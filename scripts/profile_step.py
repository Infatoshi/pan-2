"""Profile the train step at production shape; bucket CUDA kernel self-time.

Run: CUDA_VISIBLE_DEVICES=<i> uv run python scripts/profile_step.py [fwd]
Prints bucket ms/step + %, layout-kernel call count, and every distinct
kernel by self time (hand-classify with the name dump; flash kernels carry
"cutlass" in template args and defeat naive regex buckets).
"""
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from pan2.config import ModelConfig
from pan2.data.gpu_pipeline import PipelineConfig, PipelinedGpuPretrainLoader
from pan2.models.policy import PanPolicy
from pan2.train.losses import contrastive_loss
from pan2.train.speed import configure_cuda_fast_math

FWD_ONLY = len(sys.argv) > 1 and sys.argv[1] == "fwd"

configure_cuda_fast_math()
device = torch.device("cuda")
mcfg = ModelConfig(image_size=64, d_model=512, n_layers=8, n_heads=8,
                   context_len=128, action_chunk=10, n_discrete=23,
                   mouse_dim=2, backbone="transformer", frame_subsample=1)
model = PanPolicy(mcfg).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
pcfg = PipelineConfig(batch_size=32, budget_gb=8.0, num_producers=8)
loader = PipelinedGpuPretrainLoader(pcfg)


def step(b):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(b["frames"], b["goal"], b["neg"])
        loss = contrastive_loss(out["contrastive_logits"])
    if FWD_ONLY:
        return
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()


BUCKETS = [
    ("layout_nchw_nhwc", r"(?i)nchwToNhwc|nhwcToNchw|NchwToNhwc|NhwcToNchw"),
    ("gn_gelu(ours)", r"_group_norm_gelu"),
    ("adamw", r"(?i)adamw|multi_tensor"),
    ("conv_bwd", r"(?i)dgrad|wgrad"),
    ("conv_fwd", r"(?i)fprop|convolve_common|AddPadding"),
    ("conv_rest", r"(?i)cudnn|xmma|implicit_conv|convolution"),
    ("gemm", r"(?i)gemm|cutlass|nvjet|cublas|sgemm|mma|splitk|triton_tem"),
    ("attention", r"(?i)flash|fmha|sdpa|efficient_attention|memory_efficient"),
    ("inductor_pointwise", r"triton_poi_fused"),
    ("inductor_reduction", r"triton_red_fused|triton_per_fused"),
    ("softmax_loss", r"(?i)softmax|log_softmax"),
    ("memcpy_HtoD", r"(?i)memcpy hto d|memcpy h2d"),
    ("memcpy_DtoD", r"(?i)memcpy dto d|memcpy d2d"),
    ("aten_pointwise",
     r"(?i)elementwise_kernel|vectorized_elementwise|index_select|index_kernel"
     r"|catarray|gather"),
    ("groupnorm_eager", r"(?i)group_norm|RowwiseMoments|layer_norm"),
]


def bucketize(events, n_steps):
    self_ms = defaultdict(float)
    counts = defaultdict(int)
    total = 0.0
    layout_calls = 0
    top = []
    for e in events():
        if e.device_type != torch.autograd.DeviceType.CUDA or e.self_device_time_total <= 0:
            continue
        name = getattr(e, "name", None) or e.key
        ms = e.self_device_time_total / 1e3 / n_steps
        total += ms
        top.append((e.self_device_time_total / 1e3 / n_steps, name, e.count))
        for bname, pat in BUCKETS:
            if re.search(pat, name):
                self_ms[bname] += ms
                counts[bname] += e.count
                if bname == "layout_nchw_nhwc":
                    layout_calls += e.count
                break
        else:
            self_ms["other"] += ms
            counts["other"] += e.count
    return self_ms, counts, total, layout_calls, sorted(top, reverse=True)[:15]


N_PROF = 5
try:
    for _ in range(20):
        step(next(loader))
    torch.cuda.synchronize()

    # wall
    n = 60
    t0 = time.perf_counter()
    for _ in range(n):
        step(next(loader))
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / n * 1e3

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(N_PROF):
            step(next(loader))
        torch.cuda.synchronize()

    self_ms, counts, total, layout_calls, top = bucketize(prof.key_averages, N_PROF)
    mode = "FWD-ONLY" if FWD_ONLY else "FWD+BWD+OPT"
    print(f"\n=== {mode} on {torch.cuda.get_device_name(0)} ===")
    print(f"wall: {wall:.2f} ms/step | kernel self-time total: {total:.2f} "
          f"ms/step | layout calls: {layout_calls}")
    for bname, _ in BUCKETS + [("other", "")]:
        if self_ms[bname] > 1e-4:
            pct = 100 * self_ms[bname] / total
            print(f"  {bname:20s} {self_ms[bname]:7.3f} ms  {pct:5.1f}%  "
                  f"({counts[bname]} calls)")
    print("-- all distinct kernels by self time --")
    by_name = defaultdict(lambda: [0.0, 0])
    for e in prof.key_averages():
        if e.device_type != torch.autograd.DeviceType.CUDA or e.self_device_time_total <= 0:
            continue
        nm = getattr(e, "name", None) or e.key
        by_name[nm][0] += e.self_device_time_total / 1e3 / N_PROF
        by_name[nm][1] += e.count
    for nm, (ms, c) in sorted(by_name.items(), key=lambda kv: -kv[1][0]):
        if ms > 0.005:
            print(f"  {ms:7.3f} ms  x{c:4d}  {nm[:150]}")
finally:
    loader.stop()

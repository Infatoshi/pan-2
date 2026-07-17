"""Pipelined GPU-resident clip ring for pretrain.

Design
------
- Keep a large uint8 clip buffer on GPU (~budget_gb, default 10).
- Background producers: mp4/npy -> pinned host -> non_blocking H2D into free slot.
- Training thread only indexes GPU memory (no per-step disk/H2D once warm).

At 64x64, CPU ffmpeg scale is typically faster than NVDEC for short seeks; producers
still pipeline so decode overlaps with train. Swap producer backend later (DALI/NVDEC)
without changing the consumer API.
"""

from __future__ import annotations

import queue
import random
import subprocess
import tempfile
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pan2 import kernels
from pan2.data.shards import MANIFEST_NAME, load_manifest


@dataclass
class PipelineConfig:
    raw_dir: str = "/data/pan-2/raw"
    episodes_dir: str = "/data/pan-2/episodes"
    shards_dir: str = "/data/pan-2/shards"
    batch_size: int = 32
    context_len: int = 128
    frame_subsample: int = 2  # 20Hz data -> ~10fps effective tokens
    image_size: int = 64
    budget_gb: float = 10.0
    num_producers: int = 8
    prefer_source: str = "auto"  # auto | shard | mp4 | npy | pack
    device: str = "cuda"
    seed: int = 0
    # fps of the underlying source data; decode -ss timing and goal horizons
    # are computed in these native-frame units (VPT data: 20, pack refs: 10)
    native_fps: float = 20.0
    # pack corpus (custom crawl layout): index npz from scripts/build_pack_index.py
    pack_index: str = ""
    pack_minecraft_only: bool = False
    # goal = frame strictly after the context window, horizon in native frames
    # (20Hz): defaults 1s..15s into the future
    min_goal_horizon: int = 20
    max_goal_horizon: int = 300
    # same-episode hard negatives per row. 1 keeps the legacy layout (one
    # frame strictly beyond the goal window). >1 switches to wrong-horizon
    # negatives: frames from [1, 2*max_goal_horizon] after context end whose
    # horizon differs from the goal's by >= min_goal_horizon, forcing the
    # model to bind the goal to WHEN it happens, not just which episode.
    n_hard_negatives: int = 1
    # recycle a used slot with this probability so producers keep refilling
    # fresh clips/goals instead of serving a frozen capacity-sized subset.
    # Measured 2026-07-15 (shard source, 9950X3D): producers sustain ~690
    # clips/s aggregate at 8 producers (no gain at 16: per-clip CPU cost,
    # not thread count, is the limit). At Blackwell consumption (~2930
    # clips/s equivalent at 91.5 steps/s * bs 32) refresh must stay <= ~0.23
    # steady state; the default 0.05 has ~5x headroom.
    refresh_prob: float = 0.05
    # min fill fraction before yielding batches
    min_fill: float = 0.05
    # dtype the fused gather+cast emits (production trains under bf16 autocast)
    out_dtype: torch.dtype = torch.bfloat16
    # ffmpeg binary
    ffmpeg: str = "ffmpeg"
    # niceness for producer decode subprocesses. The training thread competes
    # with its own ffmpeg fleet for CPU; deprioritizing decode keeps kernel
    # launches on-core (decode has ring-buffer slack to absorb it).
    decode_nice: int = 10
    # stem-hash held-out split for per-checkpoint validation (SPEC success
    # criterion 3: "contrastive accuracy >> chance on REAL held-out clips").
    # Split by EPISODE stem, never by window: adjacent windows of one episode
    # in train and val would leak. heldout_frac=0 disables. split="train"
    # excludes held-out episodes; "val" keeps only them.
    heldout_frac: float = 0.0
    split: str = "train"


def _is_heldout(stem: str, frac: float) -> bool:
    """Deterministic episode-level membership in the held-out set.

    crc32 is stable across processes and machines, so every relaunch (incl.
    freeze-resume) sees the same split without a manifest file.
    """
    return (zlib.crc32(stem.encode()) % 10_000) / 10_000 < frac


def apply_split(items: list[dict], frac: float, split: str) -> list[dict]:
    """Filter discovered items by the held-out stem hash (see PipelineConfig)."""
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"heldout_frac must be in [0, 1), got {frac}")
    if split == "val" and frac <= 0.0:
        raise ValueError("split='val' needs heldout_frac > 0")
    if frac <= 0.0:
        return items
    keep_val = split == "val"
    out = [it for it in items if _is_heldout(it["stem"], frac) == keep_val]
    if not out:
        raise FileNotFoundError(
            f"split={split!r} with heldout_frac={frac} matched no episodes"
        )
    return out


def _subsample_indices(t: int, k: int) -> list[int]:
    k = max(1, k)
    idxs = list(range(0, t, k))
    if idxs[-1] != t - 1:
        idxs.append(t - 1)
    return idxs


class GpuClipRing:
    """Fixed GPU tensor of shape [capacity, T_sub, 3, H, W] uint8 + readiness flags."""

    def __init__(
        self,
        capacity: int,
        t_sub: int,
        image_size: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.t_sub = t_sub
        self.image_size = image_size
        self.device = device
        self.frames = torch.empty(
            capacity,
            t_sub,
            3,
            image_size,
            image_size,
            dtype=torch.uint8,
            device=device,
        )
        # host-side readiness (bool per slot); producers set under lock after H2D sync
        self.ready = [False] * capacity
        self.lock = threading.Lock()
        self.free_q: queue.Queue[int] = queue.Queue()
        self.ready_list: list[int] = []
        for i in range(capacity):
            self.free_q.put(i)
        # CUDA streams for concurrent H2D
        self.streams = [torch.cuda.Stream(device=device) for _ in range(4)]
        self._stream_i = 0

    def bytes_allocated(self) -> int:
        return self.frames.numel() * self.frames.element_size()

    def num_ready(self) -> int:
        with self.lock:
            return len(self.ready_list)

    def acquire_free(self, timeout: float | None = None) -> int | None:
        try:
            return self.free_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def publish(self, slot: int, host_uint8: torch.Tensor) -> None:
        """H2D host [T,3,H,W] uint8 into slot; mark ready."""
        stream = self.streams[self._stream_i % len(self.streams)]
        self._stream_i += 1
        # order behind default-stream work (index_select reads) since recycled
        # slots are refilled while training may still be reading them
        stream.wait_stream(torch.cuda.default_stream(device=self.device))
        with torch.cuda.stream(stream):
            self.frames[slot].copy_(host_uint8, non_blocking=True)
        stream.synchronize()
        with self.lock:
            if not self.ready[slot]:
                self.ready[slot] = True
                self.ready_list.append(slot)

    def recycle(self, slot: int) -> None:
        """Optional: free a slot for refill (LRU eviction)."""
        with self.lock:
            if self.ready[slot]:
                self.ready[slot] = False
                if slot in self.ready_list:
                    self.ready_list.remove(slot)
        self.free_q.put(slot)

    def sample_slots(self, n: int, rng: random.Random) -> list[int]:
        with self.lock:
            if len(self.ready_list) < n:
                return []
            return rng.sample(self.ready_list, n)


def _decode_mp4_window(
    mp4: Path,
    start_frame: int,
    num_frames: int,
    image_size: int,
    ffmpeg: str = "ffmpeg",
    fps: float = 20.0,
    scale: bool = True,
    nice: int = 0,
) -> np.ndarray:
    """Decode a contiguous window to uint8 NHWC RGB via ffmpeg.

    -ss before -i: seeks to the keyframe at/below the timestamp and decodes
    forward, discarding stale frames (exact-frame placement for decoding).
    scale=False skips the -vf chain entirely (source already at image_size —
    the pack ref64 layout), maximizing decode throughput.
    """
    ss = max(0.0, start_frame / fps)
    cmd = []
    if nice > 0:
        cmd += ["nice", "-n", str(nice)]
    cmd += [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        f"{ss:.3f}",
        "-i",
        str(mp4),
        "-frames:v",
        str(num_frames),
    ]
    if scale:
        cmd += ["-vf", f"scale={image_size}:{image_size}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    need = num_frames * image_size * image_size * 3
    # Unbuffered Popen + readinto instead of subprocess.run: run() drains the
    # pipe in ~32KB Python-loop chunks holding the GIL (~14% of wall measured
    # via py-spy --gil, 2026-07-17); raw readinto releases the GIL for the
    # whole read(2). stderr spools to a file so it can never backpressure.
    buf = bytearray(need)
    view = memoryview(buf)
    with tempfile.TemporaryFile(dir="/dev/shm" if Path("/dev/shm").is_dir() else None) as errf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, bufsize=0)
        got = 0
        assert proc.stdout is not None
        while got < need:
            n = proc.stdout.readinto(view[got:])
            if not n:
                break
            got += n
        proc.stdout.close()
        rc = proc.wait()
        if rc != 0 or got < need:
            errf.seek(0)
            err = errf.read()[-200:]
            raise RuntimeError(
                f"ffmpeg failed {mp4.name} rc={rc} got={got} need={need} err={err!r}"
            )
    arr = np.frombuffer(buf, dtype=np.uint8)
    return arr.reshape(num_frames, image_size, image_size, 3)


def _load_npy_window(
    img_path: Path,
    start: int,
    num_frames: int,
    image_size: int,
) -> np.ndarray:
    frames = np.load(img_path, mmap_mode="r")
    window = np.ascontiguousarray(frames[start : start + num_frames])
    # window: T,H,W,C — resize on CPU if needed (episodes are already 64)
    if window.shape[1] != image_size or window.shape[2] != image_size:
        # simple nearest via torch
        t = torch.from_numpy(window).permute(0, 3, 1, 2).float()
        t = torch.nn.functional.interpolate(
            t, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        window = t.to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
    return window


class _Producer(threading.Thread):
    def __init__(
        self,
        ring: GpuClipRing,
        items: list[dict],
        cfg: PipelineConfig,
        stop_evt: threading.Event,
        stats: dict,
        lock: threading.Lock,
    ):
        super().__init__(daemon=True)
        self.ring = ring
        self.items = items
        self.cfg = cfg
        self.stop_evt = stop_evt
        self.stats = stats
        self.lock = lock
        self.rng = random.Random(cfg.seed + id(self) % 10000)
        self._shard_cache: dict[int, np.ndarray] = {}

    def run(self) -> None:
        t_full = self.cfg.context_len
        k = self.cfg.frame_subsample
        idxs = _subsample_indices(t_full, k)
        # slot layout: [0:t_sub] context tokens, then future-goal, then negs
        assert len(idxs) + 1 + self.cfg.n_hard_negatives == self.ring.t_sub
        # One reusable pinned staging buffer per producer. Allocating pinned
        # memory per clip (cudaHostAlloc) and freeing it (cudaFreeHost)
        # device-synchronizes; at ~5 fills/step that stalled the training
        # stream ~2x (44.8 -> 21.2 ms/step with producers quiesced, GPU0
        # 2026-07-17). publish() synchronizes its copy stream before
        # returning, so the buffer is safe to overwrite on the next fill.
        self._staging = torch.empty(
            self.ring.t_sub,
            3,
            self.cfg.image_size,
            self.cfg.image_size,
            dtype=torch.uint8,
            pin_memory=True,
        )

        while not self.stop_evt.is_set():
            slot = self.ring.acquire_free(timeout=0.2)
            if slot is None:
                # ring full — brief sleep
                time.sleep(0.01)
                continue
            item = self.rng.choice(self.items)
            try:
                host = self._make_clip(item, idxs)
                # host: [t_sub+1, 3, H, W] uint8 torch pinned
                self.ring.publish(slot, host)
                with self.lock:
                    self.stats["fills"] += 1
            except Exception as e:
                # return slot and continue
                self.ring.free_q.put(slot)
                with self.lock:
                    self.stats["errors"] += 1
                    self.stats["last_error"] = repr(e)

    def _make_clip(self, item: dict, idxs: list[int]) -> torch.Tensor:
        t_full = self.cfg.context_len
        t_load = t_full + 2 * self.cfg.max_goal_horizon  # room for goal + hard neg
        src = item["source"]
        if src == "npy":
            img_path = Path(item["img"])
            frames = np.load(img_path, mmap_mode="r")
            t = int(frames.shape[0])
            if t < t_load:
                raise RuntimeError(f"short npy {img_path.name} T={t}")
            start = self.rng.randint(0, t - t_load)
            window = np.ascontiguousarray(frames[start : start + t_load])  # T,H,W,C
        elif src == "shard":
            shard = int(item["shard"])
            if shard not in self._shard_cache:
                p = Path(item["dir"]) / f"shard-{shard:05d}.frames.npy"
                self._shard_cache[shard] = np.load(p, mmap_mode="r")
            frames = self._shard_cache[shard]
            t = int(item["n_frames"])
            off = int(item["offset"])
            start = self.rng.randint(0, t - t_load)
            window = np.ascontiguousarray(frames[off + start : off + start + t_load])
        else:
            mp4 = Path(item["mp4"])
            # sample start; unknown length — use act/jsonl length if present else 0..large
            max_start = int(item.get("max_start", 5000))
            start = self.rng.randint(0, max(0, max_start))
            window = _decode_mp4_window(
                mp4,
                start_frame=start,
                num_frames=t_load,
                image_size=self.cfg.image_size,
                ffmpeg=self.cfg.ffmpeg,
                fps=self.cfg.native_fps,
                scale=int(item.get("native_size", 0)) != self.cfg.image_size,
                nice=self.cfg.decode_nice,
            )
        horizon = self.rng.randint(self.cfg.min_goal_horizon, self.cfg.max_goal_horizon)
        neg_horizons = self._sample_neg_horizons(horizon)
        # NHWC -> NCHW straight into the reusable pinned staging buffer;
        # layout [ctx toks | goal | negs...]
        t_ctx = len(idxs)
        out = self._staging
        thwc = torch.from_numpy(np.ascontiguousarray(window[idxs]))  # T,H,W,C
        out[:t_ctx].copy_(thwc.permute(0, 3, 1, 2))
        # .copy(): mp4 windows come from a read-only frombuffer view
        goal = torch.from_numpy(window[t_full - 1 + horizon].copy())  # H,W,C
        out[t_ctx].copy_(goal.permute(2, 0, 1))
        for j, h in enumerate(neg_horizons):
            n = torch.from_numpy(window[t_full - 1 + h].copy())
            out[t_ctx + 1 + j].copy_(n.permute(2, 0, 1))
        return out

    def _sample_neg_horizons(self, goal_horizon: int) -> list[int]:
        """Horizons for same-episode negatives, all within the loaded window.

        K=1 keeps the legacy scheme (strictly beyond the goal window). K>1
        draws wrong-horizon frames from [1, 2*max_goal_horizon] whose horizon
        differs from the goal's by >= min_goal_horizon (and pairwise by
        >= min separation where possible), so negatives include both
        too-early and too-late futures of the SAME episode.
        """
        cfg = self.cfg
        k = max(1, cfg.n_hard_negatives)
        if k == 1:
            off = self.rng.randint(1, cfg.max_goal_horizon)
            return [cfg.max_goal_horizon + off]
        margin = max(1, cfg.min_goal_horizon)
        chosen: list[int] = []
        tries = 0
        while len(chosen) < k and tries < 200:
            tries += 1
            h = self.rng.randint(1, 2 * cfg.max_goal_horizon)
            if abs(h - goal_horizon) < margin:
                continue
            if any(abs(h - c) < margin for c in chosen):
                continue
            chosen.append(h)
        while len(chosen) < k:  # degenerate margins: fill without pairwise rule
            h = self.rng.randint(1, 2 * cfg.max_goal_horizon)
            if abs(h - goal_horizon) >= margin:
                chosen.append(h)
        return chosen


class PipelinedGpuPretrainLoader:
    """Iterable of pretrain batches {frames, goal, neg} already on GPU.

    Slot layout at fill: [ctx toks | goal | neg], goal/neg baked per slot
    (goal 1s..15s after context end; neg past the goal window, same clip).
    Batches are emitted in cfg.out_dtype channels-last (fused gather+cast).
    """

    def __init__(self, cfg: PipelineConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("PipelinedGpuPretrainLoader requires CUDA")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = random.Random(cfg.seed)

        self.items = apply_split(
            self._discover_items(), cfg.heldout_frac, cfg.split
        )
        if not self.items:
            raise FileNotFoundError(
                f"no clips under raw={cfg.raw_dir} episodes={cfg.episodes_dir} "
                f"shards={cfg.shards_dir} pack_index={cfg.pack_index} "
                f"(split={cfg.split} heldout_frac={cfg.heldout_frac})"
            )

        idxs = _subsample_indices(cfg.context_len, cfg.frame_subsample)
        # +1 future-goal frame, +K hard-negative frames
        t_slot = len(idxs) + 1 + max(1, cfg.n_hard_negatives)
        bytes_per_clip = t_slot * 3 * cfg.image_size * cfg.image_size  # uint8
        capacity = max(64, int(cfg.budget_gb * (1024**3) // bytes_per_clip))
        # leave headroom if device has less free memory
        free, _total = torch.cuda.mem_get_info(self.device)
        max_by_free = int((free * 0.85) // bytes_per_clip)
        capacity = max(32, min(capacity, max_by_free))

        self.ring = GpuClipRing(capacity, t_slot, cfg.image_size, self.device)
        self.stats = {"fills": 0, "errors": 0, "last_error": "", "batches": 0,
                      "ready_low": capacity}
        self._stats_lock = threading.Lock()
        self._stop = threading.Event()
        self.producers = [
            _Producer(self.ring, self.items, cfg, self._stop, self.stats, self._stats_lock)
            for _ in range(cfg.num_producers)
        ]
        for p in self.producers:
            p.start()

        self._wait_for_fill()

    def _discover_items(self) -> list[dict]:
        prefer = self.cfg.prefer_source
        if prefer == "pack":
            return self._pack_items()
        shards = Path(self.cfg.shards_dir)
        if prefer in ("auto", "shard") and (shards / MANIFEST_NAME).exists():
            items = self._shard_items(shards)
            if items:
                return items
            if prefer == "shard":
                raise FileNotFoundError(
                    f"shard source requested but no usable segments under {shards}"
                )
        elif prefer == "shard":
            raise FileNotFoundError(f"shard source requested but no manifest under {shards}")

        raw = Path(self.cfg.raw_dir)
        eps = Path(self.cfg.episodes_dir)
        items: list[dict] = []

        mp4s = {p.stem: p for p in raw.glob("*.mp4")} if raw.is_dir() else {}
        imgs = {}
        if eps.is_dir():
            imgs = {p.name.replace(".img.npy", ""): p for p in eps.glob("*.img.npy")}
        stems = sorted(set(mp4s) | set(imgs))

        for stem in stems:
            has_npy = stem in imgs
            has_mp4 = stem in mp4s
            if prefer == "npy" and has_npy:
                source = "npy"
            elif prefer == "mp4" and has_mp4:
                source = "mp4"
            elif prefer == "auto":
                # npy is much cheaper fill; use it when present, else mp4
                source = "npy" if has_npy else ("mp4" if has_mp4 else "")
            else:
                source = "mp4" if has_mp4 else ("npy" if has_npy else "")
            if not source:
                continue
            rec: dict = {"stem": stem, "source": source}
            if source == "npy":
                rec["img"] = str(imgs[stem])
                try:
                    arr = np.load(imgs[stem], mmap_mode="r")
                    tlen = int(arr.shape[0])
                    t_load = self.cfg.context_len + 2 * self.cfg.max_goal_horizon
                    if tlen < t_load:
                        continue
                    rec["max_start"] = tlen - t_load
                except Exception:
                    continue
            else:
                rec["mp4"] = str(mp4s[stem])
                # pair with act npy for length if exists
                act = eps / f"{stem}.act.npy"
                if act.exists():
                    try:
                        a = np.load(act, mmap_mode="r")
                        t_load = self.cfg.context_len + 2 * self.cfg.max_goal_horizon
                        rec["max_start"] = max(0, int(a.shape[0]) - t_load)
                    except Exception:
                        rec["max_start"] = 4000
                else:
                    rec["max_start"] = 4000
            items.append(rec)
        return items

    def _pack_items(self) -> list[dict]:
        """Items from the crawl-corpus pack index (scripts/build_pack_index.py).

        The index carries exact frame counts, so max_start is real (no
        probing, no guessing like the mp4 branch). Episodes shorter than
        t_load are skipped. Index fps/image_size must match the loader
        config — the pack contract is decode-at-native, no in-pipe scaling.
        """
        idx_path = Path(self.cfg.pack_index)
        if not idx_path.is_file():
            raise FileNotFoundError(f"pack source requested but no index at {idx_path}")
        z = np.load(idx_path)
        if int(z["version"]) != 1:
            raise ValueError(f"unsupported pack index version {int(z['version'])}")
        if abs(float(z["fps"]) - self.cfg.native_fps) > 1e-6:
            raise ValueError(
                f"pack index fps {float(z['fps'])} vs cfg.native_fps {self.cfg.native_fps}"
            )
        native_size = int(z["image_size"])
        if native_size != self.cfg.image_size:
            raise ValueError(
                f"pack index is {native_size}px, loader asked {self.cfg.image_size}px "
                "(pack v1 decodes at native size, no in-pipe scaling)"
            )
        t_load = self.cfg.context_len + 2 * self.cfg.max_goal_horizon
        n_ep = len(z["stem"])
        items: list[dict] = []
        for i in range(n_ep):
            if self.cfg.pack_minecraft_only and not bool(z["minecraft"][i]):
                continue
            n = int(z["n_frames"][i])
            if n < t_load:
                continue
            items.append(
                {
                    "stem": str(z["stem"][i]),
                    "source": "pack",
                    "mp4": str(z["path"][i]),
                    "max_start": n - t_load,
                    "native_size": native_size,
                }
            )
        if not items:
            raise FileNotFoundError(
                f"empty pack items from {idx_path} "
                f"(minecraft_only={self.cfg.pack_minecraft_only}, t_load={t_load})"
            )
        return items

    def _shard_items(self, shards: Path) -> list[dict]:
        """Items from a packed shard build; segments shorter than t_load are skipped."""
        header, segments = load_manifest(shards)
        if int(header["image_size"]) != self.cfg.image_size:
            raise ValueError(
                f"shards are {header['image_size']}px, loader asked {self.cfg.image_size}"
            )
        t_load = self.cfg.context_len + 2 * self.cfg.max_goal_horizon
        items: list[dict] = []
        for seg in segments:
            t = int(seg["n_frames"])
            if t < t_load:
                continue
            items.append(
                {
                    "stem": seg["stem"],
                    "source": "shard",
                    "dir": str(shards),
                    "shard": int(seg["shard"]),
                    "offset": int(seg["offset"]),
                    "n_frames": t,
                    "max_start": t - t_load,
                }
            )
        return items

    def _wait_for_fill(self, timeout_s: float = 120.0) -> None:
        need = max(self.cfg.batch_size * 2, int(self.ring.capacity * self.cfg.min_fill))
        t0 = time.time()
        while self.ring.num_ready() < need:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(
                    f"GPU ring fill timeout: ready={self.ring.num_ready()} "
                    f"need={need} errors={self.stats['errors']} "
                    f"last={self.stats.get('last_error')}"
                )
            time.sleep(0.05)
        print(
            f"[gpu_pipeline] ready={self.ring.num_ready()}/{self.ring.capacity} "
            f"buf={self.ring.bytes_allocated()/1e9:.2f}GB "
            f"items={len(self.items)} producers={self.cfg.num_producers} "
            f"fills={self.stats['fills']} errors={self.stats['errors']}"
        )

    def stop(self) -> None:
        self._stop.set()
        for p in self.producers:
            p.join(timeout=1.0)

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        bs = self.cfg.batch_size
        # wait until enough ready
        while True:
            slots = self.ring.sample_slots(bs, self.rng)
            if slots:
                break
            if self._stop.is_set():
                raise StopIteration
            time.sleep(0.005)
        with self._stats_lock:
            # starvation observability: lowest ready count observed between
            # successful samples; if this trends toward 0 the pipeline is
            # becoming the bottleneck (see refresh_prob ceiling notes)
            self.stats["ready_low"] = min(self.stats["ready_low"], self.ring.num_ready())

        # fused: gather slots + uint8->out_dtype scaled cast + channels_last in
        # one kernel (~6x less traffic than the eager chain at production
        # shape); falls back to the torch reference composition off-CUDA.
        slot_t = torch.tensor(slots, device=self.device, dtype=torch.long)
        t_slot = self.ring.frames.shape[1]
        k_neg = max(1, self.cfg.n_hard_negatives)
        t_ctx = t_slot - 1 - k_neg
        ctx, tail = kernels.get("gather_cast")(
            self.ring.frames, slot_t, 1.0 / 255.0, self.cfg.out_dtype, t_ctx
        )
        ctx = ctx.view(bs, t_ctx, *ctx.shape[1:])
        tail = tail.view(bs, 1 + k_neg, *tail.shape[1:])
        goal = tail[:, 0]
        # single neg keeps the legacy [B,C,H,W] contract; K>1 emits [B,K,...]
        neg = tail[:, 1] if k_neg == 1 else tail[:, 1:]

        # refresh a few used slots so producers keep cycling fresh clips/goals
        for s in slots:
            if self.rng.random() < self.cfg.refresh_prob:
                self.ring.recycle(s)

        with self._stats_lock:
            self.stats["batches"] += 1
        return {"frames": ctx, "goal": goal, "neg": neg}

    def status(self) -> dict:
        return {
            "ready": self.ring.num_ready(),
            "capacity": self.ring.capacity,
            "buf_gb": self.ring.bytes_allocated() / 1e9,
            **{k: self.stats[k] for k in ("fills", "errors", "batches", "last_error", "ready_low")},
        }

"""Fused multi-tensor AdamW via a Triton pointer-table kernel.

Math matches torch.optim.AdamW(fused=True) / ATen fused_adamw (ADAMW mode):

  step is incremented before the update (float tensor on device);
  bias_correction1 = 1 - beta1**step
  bias_correction2_sqrt = sqrt(1 - beta2**step)
  p  -= lr * wd * p                 # decoupled weight decay
  m   = beta1 * m + (1 - beta1) * g
  v   = beta2 * v + (1 - beta2) * g * g
  denom = sqrt(v) / bias_correction2_sqrt + eps
  p  -= (lr / bias_correction1) * m / denom

No param repacking: each tensor keeps its storage and object identity.
Non-contiguous dense layouts (channels_last) are handled by linear storage
access over numel (elementwise; p/g/m/v share layout via zeros_like).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from pan2.kernels import register, register_reference

_TRITON_OK = False
try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

# One program covers BLOCK elements of one tensor (storage-linear).
_BLOCK = 1024


def adamw_step_ref(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    state_steps: list[torch.Tensor],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
) -> None:
    """Pure-torch AdamW step (decoupled WD), matching fused semantics."""
    if not params:
        return
    for p, g, m, v, step_t in zip(
        params, grads, exp_avgs, exp_avg_sqs, state_steps, strict=True
    ):
        step_t.add_(1)
        step = float(step_t.item())
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        bias_correction2_sqrt = bias_correction2**0.5
        if weight_decay != 0.0:
            # fused kernel: p -= lr * weight_decay * p
            p.add_(p, alpha=-lr * weight_decay)
        m.mul_(beta1).add_(g, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        step_size = lr / bias_correction1
        denom = v.sqrt().div_(bias_correction2_sqrt).add_(eps)
        p.addcdiv_(m, denom, value=-step_size)


if _TRITON_OK:

    @triton.jit
    def _adamw_ptr_table_kernel(
        p_ptrs,  # int64* [n_tensors]
        g_ptrs,
        m_ptrs,
        v_ptrs,
        numels,  # int64* [n_tensors]
        work_tid,  # int32* [n_work] tensor index per program
        work_off,  # int64* [n_work] element offset into that tensor
        step_ptr,  # fp32* scalar (already incremented)
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        BLOCK: tl.constexpr,
    ):
        wid = tl.program_id(0)
        tid = tl.load(work_tid + wid)
        start = tl.load(work_off + wid)
        n = tl.load(numels + tid)

        p_ptr = tl.load(p_ptrs + tid).to(tl.pointer_type(tl.float32))
        g_ptr = tl.load(g_ptrs + tid).to(tl.pointer_type(tl.float32))
        m_ptr = tl.load(m_ptrs + tid).to(tl.pointer_type(tl.float32))
        v_ptr = tl.load(v_ptrs + tid).to(tl.pointer_type(tl.float32))

        step = tl.load(step_ptr)
        # beta**step via exp(step * log(beta)); betas in (0,1)
        bc1 = 1.0 - tl.exp(step * tl.log(beta1))
        bc2_sqrt = tl.sqrt(1.0 - tl.exp(step * tl.log(beta2)))
        step_size = lr / bc1
        one_m_b1 = 1.0 - beta1
        one_m_b2 = 1.0 - beta2
        lr_wd = lr * weight_decay

        idx = start + tl.arange(0, BLOCK)
        mask = idx < n

        p = tl.load(p_ptr + idx, mask=mask, other=0.0)
        g = tl.load(g_ptr + idx, mask=mask, other=0.0)
        m = tl.load(m_ptr + idx, mask=mask, other=0.0)
        v = tl.load(v_ptr + idx, mask=mask, other=0.0)

        # ADAMW: p -= lr * wd * p
        p = p - lr_wd * p
        m = beta1 * m + one_m_b1 * g
        v = beta2 * v + one_m_b2 * (g * g)
        denom = tl.sqrt(v) / bc2_sqrt + eps
        p = p - step_size * (m / denom)

        tl.store(p_ptr + idx, p, mask=mask)
        tl.store(m_ptr + idx, m, mask=mask)
        tl.store(v_ptr + idx, v, mask=mask)

    def _build_work_lists(
        numels: list[int], block: int
    ) -> tuple[list[int], list[int]]:
        tids: list[int] = []
        offs: list[int] = []
        for tid, ne in enumerate(numels):
            for start in range(0, ne, block):
                tids.append(tid)
                offs.append(start)
        return tids, offs

    def adamw_step_triton(
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        state_steps: list[torch.Tensor],
        *,
        lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        eps: float,
        block: int = _BLOCK,
        cache: dict[str, Any] | None = None,
    ) -> None:
        """One multi-tensor AdamW launch over a pointer table (no repack).

        If `cache` is provided (mutable dict owned by the optimizer), stable
        tables (p/m/v ptrs, numels, work lists) are rebuilt only when the
        param set identity changes; only grad pointers refresh every step.
        """
        n = len(params)
        if n == 0:
            return
        device = params[0].device

        # bump steps first (matches torch fused: foreach_add before kernel)
        torch._foreach_add_(state_steps, 1)

        # identity key for cache invalidation
        ids = tuple(id(p) for p in params) + tuple(id(m) for m in exp_avgs)
        need_rebuild = cache is None or cache.get("ids") != ids

        if need_rebuild:
            numels = [int(p.numel()) for p in params]
            p_ptrs = torch.tensor(
                [p.data_ptr() for p in params], dtype=torch.int64, device=device
            )
            m_ptrs = torch.tensor(
                [m.data_ptr() for m in exp_avgs], dtype=torch.int64, device=device
            )
            v_ptrs = torch.tensor(
                [v.data_ptr() for v in exp_avg_sqs], dtype=torch.int64, device=device
            )
            numels_t = torch.tensor(numels, dtype=torch.int64, device=device)
            tids, offs = _build_work_lists(numels, block)
            if not tids:
                return
            work_tid = torch.tensor(tids, dtype=torch.int32, device=device)
            work_off = torch.tensor(offs, dtype=torch.int64, device=device)
            if cache is not None:
                cache.clear()
                cache.update(
                    ids=ids,
                    p_ptrs=p_ptrs,
                    m_ptrs=m_ptrs,
                    v_ptrs=v_ptrs,
                    numels_t=numels_t,
                    work_tid=work_tid,
                    work_off=work_off,
                    n_work=len(tids),
                )
        else:
            assert cache is not None
            p_ptrs = cache["p_ptrs"]
            m_ptrs = cache["m_ptrs"]
            v_ptrs = cache["v_ptrs"]
            numels_t = cache["numels_t"]
            work_tid = cache["work_tid"]
            work_off = cache["work_off"]
            n_work = cache["n_work"]
            if n_work == 0:
                return

        if cache is not None:
            n_work = cache["n_work"]
        else:
            n_work = int(work_tid.numel())

        # grads change every step under zero_grad(set_to_none=True)
        g_ptrs = torch.tensor(
            [g.data_ptr() for g in grads], dtype=torch.int64, device=device
        )

        # Shared step: all tensors in this launch are stepped together
        # (params with grad=None are omitted from the lists).
        step_ptr = state_steps[0]

        grid = (n_work,)
        _adamw_ptr_table_kernel[grid](
            p_ptrs,
            g_ptrs,
            m_ptrs,
            v_ptrs,
            numels_t,
            work_tid,
            work_off,
            step_ptr,
            float(lr),
            float(beta1),
            float(beta2),
            float(weight_decay),
            float(eps),
            BLOCK=block,
            num_warps=8,
        )

else:  # pragma: no cover

    def adamw_step_triton(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Triton not available")


def adamw_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    state_steps: list[torch.Tensor],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    cache: dict[str, Any] | None = None,
) -> None:
    """Dispatch: Triton multi-tensor on CUDA, pure-torch ref otherwise."""
    if (
        _TRITON_OK
        and params
        and params[0].is_cuda
        and params[0].dtype == torch.float32
    ):
        adamw_step_triton(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            state_steps,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            cache=cache,
        )
    else:
        adamw_step_ref(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            state_steps,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
        )


register_reference("fused_adamw", adamw_step_ref)
register("fused_adamw", adamw_step, reference=adamw_step_ref)


class FusedAdamW(Optimizer):
    """AdamW with multi-tensor Triton step; state_dict matches torch fused AdamW.

    Schema per param: step (float tensor on device), exp_avg, exp_avg_sq.
    Round-trips both ways with torch.optim.AdamW(fused=True).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        maximize: bool = False,
    ) -> None:
        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be scalar")
        lr_f = float(lr if not isinstance(lr, torch.Tensor) else lr.item())
        if not 0.0 <= lr_f:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            # mirror torch AdamW group keys for state_dict compatibility
            amsgrad=False,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=True,
            decoupled_weight_decay=True,
        )
        super().__init__(params, defaults)
        # per-group launch caches (pointer tables / work lists)
        self._launch_caches: list[dict[str, Any]] = [{} for _ in self.param_groups]

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("amsgrad", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("fused", True)
            group.setdefault("decoupled_weight_decay", True)
        self._launch_caches = [{} for _ in self.param_groups]

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        # drop non-picklable launch caches from the optimizer object path;
        # torch Optimizer.state_dict only serializes state + param_groups.
        return super().state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        super().load_state_dict(state_dict)
        self._launch_caches = [{} for _ in self.param_groups]

    @torch.no_grad()
    def step(self, closure: Any = None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if len(self._launch_caches) != len(self.param_groups):
            self._launch_caches = [{} for _ in self.param_groups]

        for gi, group in enumerate(self.param_groups):
            params: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            state_steps: list[torch.Tensor] = []

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            if isinstance(lr, torch.Tensor):
                lr = float(lr.item())
            else:
                lr = float(lr)
            weight_decay = float(group["weight_decay"])
            eps = float(group["eps"])
            maximize = bool(group.get("maximize", False))

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("FusedAdamW does not support sparse gradients")
                grad = p.grad
                if maximize:
                    grad = torch.neg(grad)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                step_t = state["step"]
                if not torch.is_tensor(step_t):
                    step_t = torch.tensor(
                        float(step_t), dtype=torch.float32, device=p.device
                    )
                    state["step"] = step_t
                elif step_t.device != p.device or step_t.dtype != torch.float32:
                    step_t = step_t.to(device=p.device, dtype=torch.float32)
                    state["step"] = step_t

                params.append(p)
                grads.append(grad)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])

            adamw_step(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                lr=lr,
                beta1=float(beta1),
                beta2=float(beta2),
                weight_decay=weight_decay,
                eps=eps,
                cache=self._launch_caches[gi],
            )
        return loss


def _env_wants_fused(default: bool = True) -> bool:
    import os

    raw = os.environ.get("PAN2_FUSED_ADAMW")
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "off", "no")


def build_adamw(
    params: Iterable[torch.nn.Parameter] | ParamsT,
    *,
    lr: float,
    weight_decay: float,
    device_type: str,
) -> Optimizer:
    """Build AdamW for training: FusedAdamW on CUDA when PAN2_FUSED_ADAMW=1."""
    if device_type == "cuda" and _env_wants_fused(default=True):
        return FusedAdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
        fused=device_type == "cuda",
    )

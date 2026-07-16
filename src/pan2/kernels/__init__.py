"""Custom kernel home for pan2.

Contract for every kernel that lands here:

1. One module per op (e.g. `pan2/kernels/frame_tokens.py`).
2. Every op ships two implementations in its module:
   - `*_ref`: pure PyTorch reference (always correct, always runnable).
   - optimized variant(s): Triton/CUDA/torch.compile, any backend.
3. `get(name)` returns the best available implementation for the current
   device, falling back to `*_ref` when the optimized path is unavailable.
   Callers (encoder, temporal, data path) never import a backend directly.
4. Every kernel has:
   - a unit test in `tests/` comparing optimized vs `*_ref` (ints bitwise,
     floats with atol/rtol), and
   - a bench in `scripts/` reporting ms against the reference at the shapes
     the model actually uses.

Speed claims in commit messages must cite the bench output. Do not land an
optimized kernel without its reference, test, and bench in the same change.
"""

from __future__ import annotations

from collections.abc import Callable

_IMPL: dict[str, Callable] = {}
_REF: dict[str, Callable] = {}


def register(name: str, impl: Callable, *, reference: Callable | None = None) -> None:
    """Register an optimized impl under `name`, optionally with its reference."""
    _IMPL[name] = impl
    if reference is not None:
        _REF[name] = reference


def register_reference(name: str, ref: Callable) -> None:
    _REF[name] = ref


def get(name: str) -> Callable:
    """Best available implementation of `name` (optimized else reference)."""
    if name in _IMPL:
        return _IMPL[name]
    if name in _REF:
        return _REF[name]
    raise KeyError(f"no kernel registered under {name!r}")


def reference(name: str) -> Callable:
    """The pure-torch reference for `name` (used by tests and benches)."""
    return _REF[name]


def available() -> list[str]:
    return sorted(set(_IMPL) | set(_REF))


def _autoload() -> None:
    """Import kernel modules so they register themselves."""
    # Local imports: each module calls register() at import time.
    from pan2.kernels import bias_gelu as _bias_gelu  # noqa: F401
    from pan2.kernels import group_norm_gelu as _group_norm_gelu  # noqa: F401
    from pan2.kernels import layer_norm_affine as _layer_norm_affine  # noqa: F401
    from pan2.kernels import residual_add as _residual_add  # noqa: F401


_autoload()

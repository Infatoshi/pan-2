# pan2.kernels

Custom op home. Rule: architecture never blocks on microbench claims, and
every optimized op lands together with its reference, its test, and its
bench. See `src/pan2/kernels/__init__.py` for the full contract.

Layout per op:

```
src/pan2/kernels/<op>.py        # *_ref + optimized + register()
tests/test_kernel_<op>.py       # optimized vs reference, bitwise or atol/rtol
scripts/bench_<op>.py           # ms at production shapes, reported in commits
```

Callers go through `pan2.kernels.get("<op>")` so backends stay swappable
(Triton, CUDA, SDPA, compile) without touching model code.

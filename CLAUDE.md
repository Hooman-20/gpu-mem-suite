# CLAUDE.md

## Ownership: read this first

**Hooman writes every CUDA kernel body and every Metal shader.** This project is how they learn
GPU memory-system programming, so the kernels are the part that has to be their own work.

Claude works only on the harness, build, scripts, tests and docs. When asked, Claude may explain
concepts (coalescing, cache hierarchy, latency hiding, `clock64`, PTX/SASS, ...) or point out
bugs in kernel code, but **Claude never writes kernel code**: no kernel bodies, no shader code,
no "here's how it could look" snippets, not even pseudo-code that maps line by line onto a kernel.
Describe the problem and let Hooman write the fix.

Files Hooman owns (Claude does not edit them):

- `cuda/kernels.cu`
- `metal/*.metal`

If a harness change needs a new kernel or a changed kernel signature, change the harness side
(`cuda/kernels.cuh`, `cuda/backend_cuda.cu`, `common/`), leave a `TODO(hooman)` stub with a
contract comment, and tell Hooman what changed.

## Layout

- `common/`: backend-agnostic host code (plain C++17, no CUDA): hash, sweep, stats, chase
  permutation, CSV, CLI, `Backend` interface, runner.
- `cuda/`: CUDA backend (`backend_cuda.cu`), entry point (`main.cu`), kernels (`kernels.cu`, Hooman's).
- `metal/`: future Metal backend + shaders (Hooman's shaders).
- `tests/`: doctest unit tests for host logic, plus `fake_backend.hpp` (CPU stand-in that follows
  the kernel contracts) for runner tests. Must never need a GPU or CUDA.
- `scripts/plot_results.py`: plots from CSVs.

## Commands

- `make test`: CPU unit tests (g++/clang only). CI runs this.
- `make` / `make smoke` / `make run ARGS="..."`: need the CUDA toolkit; smoke and run need the GPU.
- Plots: `.venv/bin/python scripts/plot_results.py` (venv: `python3 -m venv .venv && .venv/bin/pip install -r scripts/requirements.txt`).

## Conventions

- The fill pattern `pattern_word()` in `common/hash.hpp` and the `ChaseResult` layout in
  `common/kernel_abi.hpp` are part of the cross-backend contract; tests pin their values.
- Keep `common/` free of CUDA headers so the Metal backend and CI can build it.
- Environment: WSL2 Ubuntu 24.04, RTX 5090 (sm_120), CUDA 12.9 at `/usr/local/cuda`.

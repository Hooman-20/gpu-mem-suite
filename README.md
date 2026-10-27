# gpu-mem-suite

A GPU memory stress and characterization suite. It measures read, write and copy bandwidth and
load-to-use latency across working-set sizes from 4 KiB to 1 GiB, so the L1 → L2 → DRAM
transitions of a GPU's memory hierarchy show up directly in the plots. Every measurement is
checked for correctness.

The harness is backend-agnostic. CUDA runs today (RTX 5090, Blackwell `sm_120`), and a Metal
backend for Apple silicon is planned, with all backends producing the same CSV format so results
can be compared directly.

> All CUDA kernels and Metal shaders in this repository are written by hand by the author, as a
> learning exercise in GPU memory-system programming.

## What it measures

| Test    | Kernel                                        | Metric                                     |
|---------|-----------------------------------------------|--------------------------------------------|
| `read`  | streaming loads, per-thread partial sums      | GB/s                                       |
| `write` | streaming stores of a hashed pattern          | GB/s                                       |
| `copy`  | `dst[i] = src[i]`                             | GB/s (counts 2× bytes)                     |
| `chase` | single-thread random pointer chase, `clock64` | **cycles per load** (primary), ns per load |

See [docs/methodology.md](docs/methodology.md) for timing, K-selection, correctness checks and
pointer-chase design.

## Build and run

Requirements: Linux or WSL2 with the CUDA toolkit 12.8+ (`sm_120` support), GNU make, a C++17
compiler, Python 3 for plotting.

```bash
make                      # builds bin/gpumem (nvcc -O3 -std=c++17 -arch=sm_120 -lineinfo)
make test                 # host-side unit tests; no GPU or CUDA needed
make smoke                # all tests on the GPU at 4 KiB..1 MiB, quick sanity run
make run                  # full sweep, all tests
make run ARGS="--test chase --max-size 256M --chase-stride 64"
make clean
```

Use `ARCH=sm_90 make` (for example) to target a different GPU.

```text
bin/gpumem --test read|write|copy|chase|all [--min-size 4K] [--max-size 1G]
           [--points-per-octave 2] [--warmup 3] [--reps 20] [--target-ms 1.0]
           [--check-every-rep] [--seed N] [--chase-stride 128] [--block 256] ...
bin/gpumem --help         # full option list
```

Exit status: 0 if all checks passed, 2 if any size failed its correctness check, 1 on error.

## Output

Each test writes `results/<backend>_<test>_<timestamp>.csv`. The header comments record the
device (name, SM count, L2 size, clock rate, memory bus) and every run parameter. The rows give
size, K, median/min/max time, bandwidth or cycles/ns per load, and the check result.

### Plots

```bash
python3 -m venv .venv && .venv/bin/pip install -r scripts/requirements.txt
.venv/bin/python scripts/plot_results.py                       # newest run in results/
.venv/bin/python scripts/plot_results.py --ref L1=128K         # extra reference line
.venv/bin/python scripts/plot_results.py a/*.csv b/*.csv --label "RTX 5090" --label "M3 Max"
```

This produces `bandwidth_vs_size.png` and `latency_vs_size.png`. The x-axis is log2 size, with
reference lines at the L2 size read from each CSV. Rows that failed their correctness check are
marked with a red ✕.

## Results

*Placeholder: results will be added once the kernels are implemented.*

| GPU                        | Read peak (DRAM) | Write peak | Copy peak | L1 latency | L2 latency | DRAM latency |
|----------------------------|------------------|------------|-----------|------------|------------|--------------|
| NVIDIA RTX 5090 (sm_120)   | TBD              | TBD        | TBD       | TBD cycles | TBD cycles | TBD cycles   |
| Apple silicon (Metal)      | TBD              | TBD        | TBD       | TBD        | TBD        | TBD          |

<!-- ![bandwidth](docs/results/bandwidth_vs_size.png) -->
<!-- ![latency](docs/results/latency_vs_size.png) -->

## Project layout

```text
common/   backend-agnostic host code: sweep, timing loop, checks, CSV, CLI (plain C++17)
cuda/     CUDA backend and kernels
metal/    Metal backend and shaders (planned)
scripts/  plotting
tests/    host-side unit tests (doctest) + CPU fake backend for runner tests
docs/     methodology, curated results
results/  raw CSV output (git-ignored)
```

CI (GitHub Actions) builds and runs the unit tests on Ubuntu and macOS. GPU tests run locally
through `make smoke`.

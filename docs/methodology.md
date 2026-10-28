# Measurement methodology

How `gpumem` turns kernel launches into the numbers in the CSVs, and why.

## Size sweep

Buffer sizes run geometrically from `--min-size` (4 KiB) to `--max-size` (1 GiB) with
`--points-per-octave` sizes per doubling (default 2, so 37 sizes). Every size is rounded down to a
multiple of 256 bytes, so kernels may use 16-byte vector loads with no tail handling. Two points
per octave give enough resolution to locate cache-capacity transitions (L1, L2) between powers of two.

## Fill pattern

Input buffers hold `pattern_word(i, seed)`, a 32-bit integer hash of the word index and a seed
(`common/hash.hpp`). Compared with a constant fill:

- every word differs, so a kernel that reads the wrong address or skips words produces wrong output;
- the GPU memory system cannot benefit from compression or zero-page tricks;
- the host can recompute any expected word without storing a reference copy.

The function is shared verbatim by host code and CUDA kernels and will be ported bit-exactly to
Metal; unit tests pin reference values.

## Timing

For each size:

1. **Prepare** (untimed): upload inputs, compute the host-side expectation.
2. **Warm-up**: `--warmup` (3) single launches. They take cold-start effects (first-touch page
   mapping, instruction cache, clock ramp-up) out of the timed reps.
3. **Choose K**: small buffers finish in microseconds, below the useful resolution of CUDA events
   and comparable to launch overhead jitter. K launches are timed back to back as one repetition.
   K grows (at least doubling each step) until one batch lasts at least `--target-ms` (1 ms).
   Each candidate batch is timed twice and the faster run is used, because jitter only ever adds
   time. K is recorded per row in the CSV.
4. **Timed reps**: `--reps` (20) repetitions, each `cudaEventRecord` → K launches →
   `cudaEventRecord`, divided by K. The **median** is reported (robust to the occasional
   preempted rep), together with min and max as a spread indicator.
5. **Correctness check**: after the timed reps, outputs are cleared, the kernel runs once more, and
   the result is verified. `--check-every-rep` instead clears before and verifies after every
   timed repetition. The clears happen outside the timed region but do change cache state, so
   timing is slightly less representative in that mode.

Bandwidth is `bytes_per_launch / median_time`, in GB/s (10^9 bytes/s). Read and write count
`size` bytes; copy counts `2 × size` (every byte is read once and written once).

The nominal DRAM peak in the CSV header is `2 × mem_clock × bus_width / 8`, computed from the
memory clock and bus width the CUDA API reports. CUDA reports GDDR's effective clock at half the
per-pin data rate, so the factor of 2 gives the data rate. On the RTX 5090: 2 × 14.001 GHz × 512 bit / 8
= 1792 GB/s (28 Gbps GDDR7). This is a datasheet ceiling; well-written streaming kernels
typically reach about 85–92% of it.

### Known limitation: launch-bound small sizes

Back-to-back launches measure *throughput including per-launch overhead*. On this machine
(WSL2/WDDM) an empty kernel costs about 6–7 µs per launch, so sizes below a few MiB are
launch-bound rather than memory-bound, and the "bandwidth" there mostly reflects that overhead.
Ways to remove it, all candidates for future work: CUDA Graphs, doing the K repetitions
inside one launch, or using a persistent kernel.

## Correctness checks

| Test  | What the host verifies                                                            | `mismatches` counts |
|-------|------------------------------------------------------------------------------------|---------------------|
| read  | sum of all per-thread partial sums == sum of the input (mod 2^32)                  | 1 if the sums differ |
| write | every output word == `pattern_word(i, seed)`                                       | wrong words          |
| copy  | every output word == input word                                                    | wrong words          |
| chase | final index == index predicted by walking the same list on the host                | 1 if it differs      |

Outputs are zeroed (chase: set to a 0xFFFFFFFF sentinel) before the check launch, so a kernel
that writes nothing cannot pass. Any mismatch marks the row `FAIL`, and `gpumem` exits with status 2.

## Pointer chase

- Nodes sit `--chase-stride` bytes apart (default 128, one L2 line on NVIDIA), so successive
  loads never share a cache line by construction.
- The successor table is a random **single cycle** through all nodes (Sattolo's algorithm, with
  its own RNG so the permutation is identical on every platform). A single cycle guarantees the
  chase visits the whole working set rather than getting stuck in a short loop that fits in L1.
- One thread performs `steps` dependent loads (`idx = buf[idx]`), so each load's address
  depends on the previous load's data and no memory-level parallelism is possible.
- `steps` = one lap of the cycle, clamped to [`--chase-min-steps`, `--chase-max-steps`], and
  adjusted so it is never a multiple of the cycle length (otherwise the correct final index
  would equal the start index, and a kernel that does nothing would pass).
- **Primary metric:** `cycles_per_load` = `clock64()` delta around the loop / steps. This is the
  SM clock, independent of clock boosting, and comparable across GPUs.
- **Secondary:** `ns_per_load` = event time / steps. It includes launch overhead, which is
  negligible at 65k+ steps.

## CSV format

Each test writes `results/<backend>_<test>_<timestamp>.csv`. The file starts with
`# key: value` metadata (device name, SM count, L2 size, clock rate, memory clock and bus width,
CUDA versions, run parameters), followed by a normal CSV table. Columns that don't apply to a
test (bandwidth for chase, cycles for read) are left empty.

// =============================================================================================
// OWNED BY HOOMAN. Every kernel body in this file is written by hand. Claude does not write or
// edit kernel code here (see CLAUDE.md). The contracts below are what the harness checks.
//
// Guarantees from the harness for read/write/copy:
//   - Buffers come from cudaMalloc (>= 256-byte aligned).
//   - n_words is a multiple of 64 (sizes are multiples of 256 bytes), so reinterpreting the
//     buffers as uint2 / uint4 is legal with no tail handling.
//   - The grid may be smaller than n_words / blockDim.x, so a kernel must cover all of
//     [0, n_words) whatever the grid size is (e.g. a grid-stride loop).
//   - Each kernel is launched K times back to back per timed repetition and must give the same
//     result every time (no state carried between launches).
// =============================================================================================

#include "hash.hpp"
#include "kernels.cuh"

namespace gpumem {

// read_bw: measure read bandwidth.
//   Read every word of src[0, n_words) exactly once.
//   Global thread t (blockIdx.x * blockDim.x + threadIdx.x) must write
//       partials[t] = wrapping uint32 sum of every word thread t loaded   (0 if it loaded none).
//   The host checks sum(partials) == sum(src) mod 2^32. Writing the sums is what keeps the
//   compiler from deleting the loads; keep the extra work per word small (one add).
__global__ void read_bw(const uint32_t* __restrict__ src, uint64_t n_words, uint32_t* __restrict__ partials) {
  // TODO(hooman): implement read_bw
}

// write_bw: measure write bandwidth.
//   For every i in [0, n_words): dst[i] = gpumem::pattern_word(i, seed)  (common/hash.hpp).
//   Do not read dst. The host zeroes dst before the check launch and compares every word.
//   Note: the hash costs a handful of integer ops per word; whether that limits write
//   bandwidth is worth measuring.
__global__ void write_bw(uint32_t* __restrict__ dst, uint64_t n_words, uint32_t seed) {
  // TODO(hooman): implement write_bw
}

// copy_bw: measure copy bandwidth (the harness counts 2 * n_words * 4 bytes per launch).
//   For every i in [0, n_words): dst[i] = src[i]. src holds pattern_word(i, seed); the host
//   zeroes dst before the check launch and compares every word.
__global__ void copy_bw(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, uint64_t n_words) {
  // TODO(hooman): implement copy_bw
}

// pointer_chase: measure load-to-use latency. Launched as a single thread (<<<1, 1>>>).
//   buf is a random single-cycle linked list of word indices (built by common/chase.cpp):
//   idx = start; repeat `steps` times: idx = buf[idx].
//   Read clock64() right before and right after the dependent-load loop and write
//       out->cycles      = end - begin
//       out->final_index = idx
//   The host walks the same list and flags a mismatch if final_index differs. The runner
//   reports out->cycles / steps as cycles per load (primary metric) and event time / steps
//   as ns per load.
__global__ void pointer_chase(const uint32_t* __restrict__ buf, uint32_t start, uint64_t steps,
                              ChaseResult* __restrict__ out) {
  // TODO(hooman): implement pointer_chase
}

// ---- launch glue: forwards the harness launch config. Plain host code, no kernel logic. ----

void launch_read_bw(LaunchConfig cfg, cudaStream_t stream, const uint32_t* src, uint64_t n_words,
                    uint32_t* partials) {
  read_bw<<<cfg.grid, cfg.block, 0, stream>>>(src, n_words, partials);
}

void launch_write_bw(LaunchConfig cfg, cudaStream_t stream, uint32_t* dst, uint64_t n_words, uint32_t seed) {
  write_bw<<<cfg.grid, cfg.block, 0, stream>>>(dst, n_words, seed);
}

void launch_copy_bw(LaunchConfig cfg, cudaStream_t stream, const uint32_t* src, uint32_t* dst, uint64_t n_words) {
  copy_bw<<<cfg.grid, cfg.block, 0, stream>>>(src, dst, n_words);
}

void launch_pointer_chase(cudaStream_t stream, const uint32_t* buf, uint32_t start, uint64_t steps,
                          ChaseResult* out) {
  pointer_chase<<<1, 1, 0, stream>>>(buf, start, steps, out);
}

}  // namespace gpumem

#pragma once
// Interface between the CUDA backend (harness) and the hand-written kernels in kernels.cu.

#include <cstdint>
#include <cuda_runtime.h>

#include "kernel_abi.hpp"

namespace gpumem {

struct LaunchConfig {
  dim3 grid;
  dim3 block;
};

// Streaming kernels get grid = min(ceil(n_words / block), SMs * blocks_per_sm) blocks of
// `block` threads (see --block / --blocks-per-cu), so the grid can be smaller than the data.
void launch_read_bw(LaunchConfig cfg, cudaStream_t stream, const uint32_t* src, uint64_t n_words,
                    uint32_t* partials);
void launch_write_bw(LaunchConfig cfg, cudaStream_t stream, uint32_t* dst, uint64_t n_words, uint32_t seed);
void launch_copy_bw(LaunchConfig cfg, cudaStream_t stream, const uint32_t* src, uint32_t* dst, uint64_t n_words);

// Always one thread.
void launch_pointer_chase(cudaStream_t stream, const uint32_t* buf, uint32_t start, uint64_t steps,
                          ChaseResult* out);

}  // namespace gpumem

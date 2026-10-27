#pragma once
// Data layouts written by GPU kernels and read by the host. Every backend (CUDA, Metal)
// must produce exactly these layouts.

#include <cstdint>

namespace gpumem {

// Output of one pointer-chase launch.
struct ChaseResult {
  uint32_t final_index;  // word index reached after `steps` dependent loads
  uint32_t reserved;     // keeps `cycles` 8-byte aligned on every backend
  uint64_t cycles;       // cycle-counter delta around the dependent-load loop
};
static_assert(sizeof(ChaseResult) == 16, "ChaseResult layout is part of the kernel ABI");

}  // namespace gpumem

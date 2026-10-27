#pragma once
// Fill pattern shared by host code and every GPU backend.
// Must stay bit-identical across CUDA, Metal and host; tests/test_hash.cpp pins reference values
// so a Metal port can be checked against the same numbers.

#include <cstdint>

#if defined(__CUDACC__)
#define GPUMEM_HD __host__ __device__ __forceinline__
#else
#define GPUMEM_HD inline
#endif

namespace gpumem {

// "lowbias32" integer finalizer (Chris Wellons, public domain). Bijective on uint32.
GPUMEM_HD uint32_t hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// Word stored at 32-bit word `index` of a buffer filled with `seed`.
// The golden-ratio constant keeps (index 0, seed 0) from mapping to 0, so an untouched
// zeroed buffer does not look like valid output at the start of the buffer.
GPUMEM_HD uint32_t pattern_word(uint64_t index, uint32_t seed) {
  const uint32_t lo = static_cast<uint32_t>(index);
  const uint32_t hi = static_cast<uint32_t>(index >> 32);
  return hash32(lo ^ hash32(hi ^ seed ^ 0x9e3779b9U));
}

}  // namespace gpumem

#pragma once
// Host side of the pointer-chase test: building the chase buffer and predicting where the
// GPU's chase must end up.

#include <cstdint>
#include <vector>

namespace gpumem {

// Random permutation forming one cycle through all n nodes (Sattolo's algorithm), as a
// successor table: next[i] is the node visited after node i. Uses its own RNG so the same
// seed gives the same permutation on every platform and standard library.
std::vector<uint32_t> make_cyclic_permutation(uint32_t n, uint64_t seed);

// True when `next` is a permutation of [0, n) consisting of a single n-cycle.
bool is_single_cycle(const std::vector<uint32_t>& next);

// Lays nodes out `stride_words` apart: buf[i * stride_words] = next[i] * stride_words, so the
// kernel only has to do `idx = buf[idx]`. Words between nodes are 0.
std::vector<uint32_t> build_chase_buffer(const std::vector<uint32_t>& next, uint32_t stride_words);

// Word index reached after `steps` dependent loads starting at word index `start`.
uint32_t predict_chase_final(const std::vector<uint32_t>& buf, uint32_t start, uint64_t steps);

// Dependent loads per launch. `fixed_steps` > 0 wins; otherwise one full lap of the cycle,
// clamped to [min_steps, max_steps]. Never returns a multiple of a cycle length > 1, so a
// kernel that skips the chase and reports `start` cannot pass the check.
uint64_t choose_chase_steps(uint64_t n_nodes, uint64_t fixed_steps, uint64_t min_steps, uint64_t max_steps);

}  // namespace gpumem

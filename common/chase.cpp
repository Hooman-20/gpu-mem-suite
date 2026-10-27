#include "chase.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace gpumem {
namespace {

uint64_t splitmix64(uint64_t& state) {
  uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

}  // namespace

std::vector<uint32_t> make_cyclic_permutation(uint32_t n, uint64_t seed) {
  if (n == 0) throw std::invalid_argument("permutation needs at least one node");
  std::vector<uint32_t> next(n);
  std::iota(next.begin(), next.end(), 0U);
  uint64_t state = seed;
  for (uint32_t i = n - 1; i > 0; --i) {
    // j < i (never j == i) is what makes Sattolo produce a single cycle. The modulo bias is
    // below 2^-32 for any n we can allocate.
    const auto j = static_cast<uint32_t>(splitmix64(state) % i);
    std::swap(next[i], next[j]);
  }
  return next;
}

bool is_single_cycle(const std::vector<uint32_t>& next) {
  const size_t n = next.size();
  if (n == 0) return false;
  std::vector<bool> seen(n, false);
  size_t node = 0;
  for (size_t step = 0; step < n; ++step) {
    if (seen[node]) return false;
    seen[node] = true;
    if (next[node] >= n) return false;
    node = next[node];
  }
  return node == 0;
}

std::vector<uint32_t> build_chase_buffer(const std::vector<uint32_t>& next, uint32_t stride_words) {
  if (stride_words == 0) throw std::invalid_argument("chase stride must be non-zero");
  const uint64_t words = static_cast<uint64_t>(next.size()) * stride_words;
  if (words > (uint64_t{1} << 32)) throw std::invalid_argument("chase buffer exceeds 32-bit word indices");
  std::vector<uint32_t> buf(static_cast<size_t>(words), 0U);
  for (size_t i = 0; i < next.size(); ++i) buf[i * stride_words] = next[i] * stride_words;
  return buf;
}

uint32_t predict_chase_final(const std::vector<uint32_t>& buf, uint32_t start, uint64_t steps) {
  uint32_t idx = start;
  for (uint64_t s = 0; s < steps; ++s) {
    if (idx >= buf.size()) throw std::out_of_range("chase walked outside the buffer");
    idx = buf[idx];
  }
  return idx;
}

uint64_t choose_chase_steps(uint64_t n_nodes, uint64_t fixed_steps, uint64_t min_steps, uint64_t max_steps) {
  if (n_nodes == 0) throw std::invalid_argument("chase needs at least one node");
  uint64_t steps = fixed_steps > 0 ? fixed_steps : std::clamp(n_nodes, min_steps, max_steps);
  if (n_nodes > 1 && steps % n_nodes == 0) ++steps;
  return steps;
}

}  // namespace gpumem

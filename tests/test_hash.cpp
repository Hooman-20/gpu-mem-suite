#include "doctest.h"
#include "hash.hpp"

#include <set>

using namespace gpumem;

TEST_CASE("hash32 matches pinned reference values") {
  // Any backend (CUDA, Metal) must reproduce these exactly.
  CHECK(hash32(0) == 0x00000000U);
  CHECK(hash32(1) == 0x688990c0U);
  CHECK(hash32(0xdeadbeefU) == 0xe628c683U);
}

TEST_CASE("pattern_word matches pinned reference values") {
  CHECK(pattern_word(0, 0) == 0xae6f80f1U);
  CHECK(pattern_word(1, 0) == 0xa07c7a97U);
  CHECK(pattern_word(2, 0) == 0x0e77ceb6U);
  CHECK(pattern_word(1000, 0) == 0xd6663a0cU);
  CHECK(pattern_word(0, 0x5eed1234U) == 0x95b827f2U);
  CHECK(pattern_word((1ULL << 28) - 1, 0x5eed1234U) == 0xcccdfceaU);  // last word of a 1 GiB buffer
  CHECK(pattern_word((1ULL << 32) + 5, 0x5eed1234U) == 0x22a64de7U);  // high index bits matter
}

TEST_CASE("pattern_word is deterministic and seed-dependent") {
  CHECK(pattern_word(12345, 7) == pattern_word(12345, 7));
  CHECK(pattern_word(12345, 7) != pattern_word(12345, 8));
  CHECK(pattern_word(5, 1) != pattern_word(5 + (1ULL << 32), 1));
}

TEST_CASE("pattern_word has no collisions over a small range and is not zero at index 0") {
  std::set<uint32_t> seen;
  for (uint64_t i = 0; i < 100000; ++i) seen.insert(pattern_word(i, 42));
  CHECK(seen.size() == 100000);  // hash32 is a bijection, so a fixed seed cannot collide
  for (uint32_t seed : {0U, 1U, 0x5eed1234U}) CHECK(pattern_word(0, seed) != 0U);
}

TEST_CASE("pattern_word bits are roughly balanced") {
  uint64_t ones = 0;
  const int n = 1 << 16;
  for (int i = 0; i < n; ++i) ones += __builtin_popcount(pattern_word(static_cast<uint64_t>(i), 3));
  const double fraction = static_cast<double>(ones) / (32.0 * n);
  CHECK(fraction == doctest::Approx(0.5).epsilon(0.01));
}

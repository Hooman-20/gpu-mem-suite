#include "chase.hpp"
#include "doctest.h"

#include <numeric>
#include <stdexcept>

using namespace gpumem;

TEST_CASE("cyclic permutation is a single full cycle for many sizes and seeds") {
  for (uint32_t n : {1U, 2U, 3U, 7U, 32U, 1000U, 65537U}) {
    for (uint64_t seed : {0ULL, 1ULL, 0xdeadbeefULL}) {
      CAPTURE(n);
      CAPTURE(seed);
      CHECK(is_single_cycle(make_cyclic_permutation(n, seed)));
    }
  }
}

TEST_CASE("cyclic permutation of a cache-busting size is a single cycle") {
  // 1 GiB / 128 B nodes, the largest default chase.
  CHECK(is_single_cycle(make_cyclic_permutation(1U << 23, 0x5eed1234ULL)));
}

TEST_CASE("cyclic permutation has no fixed points") {
  const auto next = make_cyclic_permutation(5000, 9);
  for (uint32_t i = 0; i < next.size(); ++i) CHECK(next[i] != i);
}

TEST_CASE("cyclic permutation is deterministic per seed") {
  CHECK(make_cyclic_permutation(1000, 5) == make_cyclic_permutation(1000, 5));
  CHECK(make_cyclic_permutation(1000, 5) != make_cyclic_permutation(1000, 6));
}

TEST_CASE("cyclic permutation is not trivially sequential") {
  const auto next = make_cyclic_permutation(4096, 77);
  int sequential = 0;
  for (uint32_t i = 0; i < next.size(); ++i) sequential += next[i] == (i + 1) % next.size();
  CHECK(sequential < 16);  // a prefetcher-friendly i -> i+1 walk would defeat the test
}

TEST_CASE("cyclic permutation rejects n == 0") {
  CHECK_THROWS_AS(make_cyclic_permutation(0, 1), std::invalid_argument);
}

TEST_CASE("is_single_cycle rejects non-cycles") {
  std::vector<uint32_t> identity(8);
  std::iota(identity.begin(), identity.end(), 0U);
  CHECK_FALSE(is_single_cycle(identity));
  CHECK_FALSE(is_single_cycle({1, 0, 3, 2}));  // two 2-cycles
  CHECK_FALSE(is_single_cycle({1, 2, 1}));     // not a permutation
  CHECK_FALSE(is_single_cycle({1, 5}));        // out of range
  CHECK_FALSE(is_single_cycle({}));
  CHECK(is_single_cycle({1, 2, 0}));
}

TEST_CASE("chase buffer stores scaled successors at the stride") {
  const std::vector<uint32_t> next = {2, 0, 1};
  const auto buf = build_chase_buffer(next, 32);
  REQUIRE(buf.size() == 96);
  CHECK(buf[0] == 64);
  CHECK(buf[32] == 0);
  CHECK(buf[64] == 32);
  CHECK(buf[1] == 0);
  CHECK(buf[95] == 0);
}

TEST_CASE("predicted final index walks the chain") {
  const auto next = make_cyclic_permutation(100, 3);
  const auto buf = build_chase_buffer(next, 4);
  CHECK(predict_chase_final(buf, 0, 0) == 0);
  CHECK(predict_chase_final(buf, 0, 1) == next[0] * 4);
  CHECK(predict_chase_final(buf, 0, 2) == next[next[0]] * 4);
  CHECK(predict_chase_final(buf, 0, 100) == 0);  // one full lap returns to the start
  CHECK(predict_chase_final(buf, 0, 101) == next[0] * 4);
}

TEST_CASE("predicted chase never ends at the start for a cycle longer than one node") {
  for (uint64_t n_nodes : {2ULL, 32ULL, 1000ULL, 1ULL << 16, 1ULL << 20}) {
    const uint64_t steps = choose_chase_steps(n_nodes, 0, 1ULL << 16, 1ULL << 18);
    CAPTURE(n_nodes);
    CHECK(steps % n_nodes != 0);
  }
}

TEST_CASE("choose_chase_steps clamps one lap and honours a fixed count") {
  CHECK(choose_chase_steps(1000, 0, 100, 10000) == 1001);       // one lap would return to start
  CHECK(choose_chase_steps(10, 0, 100, 10000) == 101);          // raised to min, then off the lap
  CHECK(choose_chase_steps(7, 0, 100, 10000) == 100);           // raised to min
  CHECK(choose_chase_steps(1000000, 0, 100, 10000) == 10000);   // capped at max
  CHECK(choose_chase_steps(1000, 777, 100, 10000) == 777);      // fixed wins
  CHECK(choose_chase_steps(1, 0, 100, 10000) == 100);           // self-loop: nothing to avoid
  CHECK_THROWS_AS(choose_chase_steps(0, 0, 1, 2), std::invalid_argument);
}

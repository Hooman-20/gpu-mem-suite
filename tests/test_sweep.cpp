#include "doctest.h"
#include "sweep.hpp"

#include <stdexcept>

using namespace gpumem;

TEST_CASE("default sweep is 4 KiB to 1 GiB in powers of two") {
  const auto sizes = make_size_sweep(4 * KiB, 1 * GiB, 1, 256);
  REQUIRE(sizes.size() == 19);
  CHECK(sizes.front() == 4 * KiB);
  CHECK(sizes.back() == 1 * GiB);
  for (size_t i = 1; i < sizes.size(); ++i) CHECK(sizes[i] == 2 * sizes[i - 1]);
}

TEST_CASE("two points per octave interleaves aligned midpoints") {
  const auto sizes = make_size_sweep(4 * KiB, 1 * GiB, 2, 256);
  REQUIRE(sizes.size() == 37);
  CHECK(sizes.front() == 4 * KiB);
  CHECK(sizes[1] == 5632);  // 4096 * sqrt(2) = 5792.6, aligned down to a multiple of 256
  CHECK(sizes.back() == 1 * GiB);
  for (size_t i = 0; i < sizes.size(); ++i) {
    CHECK(sizes[i] % 256 == 0);
    if (i) CHECK(sizes[i] > sizes[i - 1]);
  }
}

TEST_CASE("sweep ends at an off-grid maximum") {
  const auto sizes = make_size_sweep(4 * KiB, 10 * KiB, 1, 256);
  REQUIRE(sizes.size() == 3);
  CHECK(sizes[0] == 4 * KiB);
  CHECK(sizes[1] == 8 * KiB);
  CHECK(sizes[2] == 10 * KiB);
}

TEST_CASE("sweep with min == max is a single size") {
  const auto sizes = make_size_sweep(64 * KiB, 64 * KiB, 4, 256);
  REQUIRE(sizes.size() == 1);
  CHECK(sizes[0] == 64 * KiB);
}

TEST_CASE("sweep rejects bad arguments") {
  CHECK_THROWS_AS(make_size_sweep(8 * KiB, 4 * KiB, 1, 256), std::invalid_argument);
  CHECK_THROWS_AS(make_size_sweep(128, 4 * KiB, 1, 256), std::invalid_argument);
  CHECK_THROWS_AS(make_size_sweep(4 * KiB, 8 * KiB, 0, 256), std::invalid_argument);
  CHECK_THROWS_AS(make_size_sweep(4 * KiB, 8 * KiB, 1, 0), std::invalid_argument);
}

TEST_CASE("choose_k makes one repetition last at least the target") {
  CHECK(choose_k(2.0, 1.0, 10000) == 1);     // already long enough
  CHECK(choose_k(1.0, 1.0, 10000) == 1);
  CHECK(choose_k(0.3, 1.0, 10000) == 4);     // ceil(3.33)
  CHECK(choose_k(0.25, 1.0, 10000) == 4);    // exact division
  CHECK(choose_k(0.001, 1.0, 10000) == 1000);
  for (double ms : {0.0007, 0.013, 0.2, 0.9}) CHECK(choose_k(ms, 1.0, 1000000) * ms >= 1.0);
}

TEST_CASE("choose_k clamps and handles degenerate timings") {
  CHECK(choose_k(0.00001, 1.0, 500) == 500);
  CHECK(choose_k(0.0, 1.0, 500) == 500);
  CHECK(choose_k(-1.0, 1.0, 500) == 500);
  CHECK(choose_k(0.0 / 0.0, 1.0, 500) == 500);
  CHECK_THROWS_AS(choose_k(1.0, 1.0, 0), std::invalid_argument);
}

TEST_CASE("parse_size understands binary units") {
  CHECK(parse_size("4096") == 4096);
  CHECK(parse_size("512B") == 512);
  CHECK(parse_size("4K") == 4 * KiB);
  CHECK(parse_size("4kb") == 4 * KiB);
  CHECK(parse_size("4KiB") == 4 * KiB);
  CHECK(parse_size("256M") == 256 * MiB);
  CHECK(parse_size("1G") == GiB);
  CHECK(parse_size("2GiB") == 2 * GiB);
}

TEST_CASE("parse_size rejects garbage") {
  CHECK_THROWS_AS(parse_size(""), std::invalid_argument);
  CHECK_THROWS_AS(parse_size("K"), std::invalid_argument);
  CHECK_THROWS_AS(parse_size("4X"), std::invalid_argument);
  CHECK_THROWS_AS(parse_size("-4K"), std::invalid_argument);
  CHECK_THROWS_AS(parse_size("1.5M"), std::invalid_argument);
  CHECK_THROWS_AS(parse_size("99999999999999999999"), std::invalid_argument);
  CHECK_THROWS_AS(parse_size("99999999999G"), std::invalid_argument);
}

TEST_CASE("format_size") {
  CHECK(format_size(100) == "100 B");
  CHECK(format_size(4096) == "4 KiB");
  CHECK(format_size(5888) == "5.75 KiB");
  CHECK(format_size(96 * MiB) == "96 MiB");
  CHECK(format_size(GiB) == "1 GiB");
}

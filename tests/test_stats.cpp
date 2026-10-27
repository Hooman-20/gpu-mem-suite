#include "doctest.h"
#include "stats.hpp"

#include <stdexcept>

using namespace gpumem;

TEST_CASE("median of odd-length input is the middle value") {
  CHECK(median({3.0}) == 3.0);
  CHECK(median({5.0, 1.0, 3.0}) == 3.0);
  CHECK(median({9.0, 2.0, 7.0, 4.0, 100.0}) == 7.0);
}

TEST_CASE("median of even-length input averages the two middle values") {
  CHECK(median({1.0, 2.0}) == 1.5);
  CHECK(median({4.0, 1.0, 3.0, 2.0}) == 2.5);
}

TEST_CASE("median is robust to an outlier") {
  CHECK(median({1.0, 1.1, 0.9, 1.0, 50.0}) == 1.0);
}

TEST_CASE("median handles duplicates") {
  CHECK(median({2.0, 2.0, 2.0, 2.0}) == 2.0);
}

TEST_CASE("median of empty input throws") {
  CHECK_THROWS_AS(median({}), std::invalid_argument);
}

TEST_CASE("summarize reports median, min and max") {
  const Summary s = summarize({0.4, 0.1, 0.3, 0.2, 0.5});
  CHECK(s.median == 0.3);
  CHECK(s.min == 0.1);
  CHECK(s.max == 0.5);
}

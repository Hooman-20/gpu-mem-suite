#include "backend.hpp"
#include "doctest.h"

#include <stdexcept>

using namespace gpumem;

TEST_CASE("nominal DRAM bandwidth from memory clock and bus width") {
  // RTX 5090 as reported by CUDA: 14.001 GHz, 512-bit GDDR7 -> 28 Gbps/pin -> ~1792 GB/s.
  CHECK(nominal_dram_gbs(14001000, 512) == doctest::Approx(1792.128));
  // RTX 4090: 10.501 GHz, 384-bit GDDR6X -> ~1008 GB/s.
  CHECK(nominal_dram_gbs(10501000, 384) == doctest::Approx(1008.096));
  CHECK(nominal_dram_gbs(0, 512) == 0.0);
  CHECK(nominal_dram_gbs(14001000, 0) == 0.0);
}

TEST_CASE("test names round-trip through the parser") {
  for (Test t : {Test::Read, Test::Write, Test::Copy, Test::Chase}) {
    CHECK(parse_test_list(test_name(t)) == std::vector<Test>{t});
  }
  CHECK_THROWS_AS(parse_test_list(""), std::invalid_argument);
}

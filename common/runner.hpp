#pragma once
// Backend-agnostic measurement loop: sweep sizes, warm up, pick K, time reps, verify, write CSV.

#include <ostream>
#include <string>
#include <vector>

#include "backend.hpp"
#include "cli.hpp"

namespace gpumem {

// Every swept size is a multiple of this, so buffers can always be read as uint4 / 16-byte vectors.
constexpr uint64_t kSizeAlign = 256;

// CSV columns, in order. Empty fields mean "not applicable for this test".
extern const std::vector<std::string> kCsvColumns;

struct SuiteResult {
  int failed_sizes = 0;                // sizes whose correctness check reported mismatches
  std::vector<std::string> csv_paths;  // one file per test
};

SuiteResult run_suite(Backend& backend, const Options& opts, std::ostream& log);

}  // namespace gpumem

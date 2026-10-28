#include "backend.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace gpumem {

const char* test_name(Test test) {
  switch (test) {
    case Test::Read: return "read";
    case Test::Write: return "write";
    case Test::Copy: return "copy";
    case Test::Chase: return "chase";
  }
  return "unknown";
}

const char* timing_mode_name(TimingMode mode) {
  return mode == TimingMode::Graph ? "graph" : "launch";
}

TimingMode parse_timing_mode(const std::string& text) {
  if (text == "launch") return TimingMode::Launch;
  if (text == "graph") return TimingMode::Graph;
  throw std::invalid_argument("unknown timing mode '" + text + "' (launch|graph)");
}

std::vector<Test> parse_test_list(const std::string& text) {
  static const Test kAll[] = {Test::Read, Test::Write, Test::Copy, Test::Chase};
  std::vector<Test> tests;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item == "all") {
      for (Test t : kAll) {
        if (std::find(tests.begin(), tests.end(), t) == tests.end()) tests.push_back(t);
      }
      continue;
    }
    const auto* match = std::find_if(std::begin(kAll), std::end(kAll), [&](Test t) { return item == test_name(t); });
    if (match == std::end(kAll)) throw std::invalid_argument("unknown test '" + item + "' (read|write|copy|chase|all)");
    if (std::find(tests.begin(), tests.end(), *match) == tests.end()) tests.push_back(*match);
  }
  if (tests.empty()) throw std::invalid_argument("no tests selected");
  return tests;
}

double nominal_dram_gbs(uint64_t mem_clock_khz, int bus_width_bits) {
  if (mem_clock_khz == 0 || bus_width_bits <= 0) return 0.0;
  const double transfers_per_s = 2.0 * static_cast<double>(mem_clock_khz) * 1e3;
  return transfers_per_s * (bus_width_bits / 8.0) / 1e9;
}

}  // namespace gpumem

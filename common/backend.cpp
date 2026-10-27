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

}  // namespace gpumem

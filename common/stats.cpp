#include "stats.hpp"

#include <algorithm>
#include <stdexcept>

namespace gpumem {

double median(std::vector<double> values) {
  if (values.empty()) throw std::invalid_argument("median of an empty set");
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  const double upper = values[mid];
  if (values.size() % 2 == 1) return upper;
  const double lower = *std::max_element(values.begin(), values.begin() + mid);
  return (lower + upper) / 2.0;
}

Summary summarize(const std::vector<double>& values) {
  Summary s;
  s.median = median(values);
  const auto [lo, hi] = std::minmax_element(values.begin(), values.end());
  s.min = *lo;
  s.max = *hi;
  return s;
}

}  // namespace gpumem

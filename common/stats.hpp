#pragma once

#include <vector>

namespace gpumem {

struct Summary {
  double median = 0.0;
  double min = 0.0;
  double max = 0.0;
};

// Median of `values` (mean of the two middle values for even counts). Throws on empty input.
double median(std::vector<double> values);

Summary summarize(const std::vector<double>& values);

}  // namespace gpumem

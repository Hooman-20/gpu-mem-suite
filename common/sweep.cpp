#include "sweep.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace gpumem {

std::vector<uint64_t> make_size_sweep(uint64_t min_bytes, uint64_t max_bytes, int points_per_octave,
                                      uint64_t align) {
  if (align == 0) throw std::invalid_argument("sweep alignment must be non-zero");
  if (points_per_octave < 1) throw std::invalid_argument("points per octave must be >= 1");
  if (min_bytes < align) throw std::invalid_argument("minimum size is smaller than the alignment");
  if (min_bytes > max_bytes) throw std::invalid_argument("minimum size is larger than maximum size");

  const uint64_t last = max_bytes / align * align;
  std::vector<uint64_t> sizes;
  for (int i = 0;; ++i) {
    const double exact = static_cast<double>(min_bytes) * std::exp2(static_cast<double>(i) / points_per_octave);
    if (exact > static_cast<double>(max_bytes) * (1.0 + 1e-12)) break;
    uint64_t size = static_cast<uint64_t>(std::llround(exact)) / align * align;
    if (size > last) size = last;
    if (sizes.empty() || size > sizes.back()) sizes.push_back(size);
  }
  if (sizes.back() < last) sizes.push_back(last);
  return sizes;
}

int choose_k(double single_launch_ms, double target_ms, int max_k) {
  if (max_k < 1) throw std::invalid_argument("max_k must be >= 1");
  if (!(single_launch_ms > 0.0)) return max_k;
  const double k = std::ceil(target_ms / single_launch_ms);
  if (k < 1.0) return 1;
  if (k > static_cast<double>(max_k)) return max_k;
  return static_cast<int>(k);
}

uint64_t parse_size(const std::string& text) {
  size_t pos = 0;
  uint64_t value = 0;
  while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
    const uint64_t digit = static_cast<uint64_t>(text[pos] - '0');
    if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
      throw std::invalid_argument("size out of range: " + text);
    }
    value = value * 10 + digit;
    ++pos;
  }
  if (pos == 0) throw std::invalid_argument("invalid size: '" + text + "'");

  std::string unit;
  for (size_t i = pos; i < text.size(); ++i) unit += static_cast<char>(std::tolower(static_cast<unsigned char>(text[i])));

  uint64_t scale = 0;
  if (unit.empty() || unit == "b") scale = 1;
  else if (unit == "k" || unit == "kb" || unit == "kib") scale = KiB;
  else if (unit == "m" || unit == "mb" || unit == "mib") scale = MiB;
  else if (unit == "g" || unit == "gb" || unit == "gib") scale = GiB;
  else throw std::invalid_argument("invalid size unit in '" + text + "'");

  if (value > std::numeric_limits<uint64_t>::max() / scale) throw std::invalid_argument("size out of range: " + text);
  return value * scale;
}

std::string format_size(uint64_t bytes) {
  static const char* const kUnits[] = {"B", "KiB", "MiB", "GiB"};
  int unit = 0;
  uint64_t scale = 1;
  while (unit < 3 && bytes >= scale * 1024) {
    scale *= 1024;
    ++unit;
  }
  char buf[32];
  if (bytes % scale == 0) {
    std::snprintf(buf, sizeof buf, "%llu %s", static_cast<unsigned long long>(bytes / scale), kUnits[unit]);
  } else {
    std::snprintf(buf, sizeof buf, "%.2f %s", static_cast<double>(bytes) / static_cast<double>(scale), kUnits[unit]);
  }
  return buf;
}

}  // namespace gpumem

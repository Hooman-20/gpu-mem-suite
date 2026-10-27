#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gpumem {

constexpr uint64_t KiB = 1024;
constexpr uint64_t MiB = 1024 * KiB;
constexpr uint64_t GiB = 1024 * MiB;

// Geometric sweep from min_bytes to max_bytes with `points_per_octave` sizes per doubling.
// Every size is rounded down to a multiple of `align`; the result is strictly increasing and
// always ends at max_bytes (rounded down to `align`).
std::vector<uint64_t> make_size_sweep(uint64_t min_bytes, uint64_t max_bytes, int points_per_octave,
                                      uint64_t align);

// Launches per timed repetition so one repetition lasts at least `target_ms`.
// `single_launch_ms` is a measured warm launch; non-positive or NaN timings give max_k.
int choose_k(double single_launch_ms, double target_ms, int max_k);

// "4096", "4K", "4KB", "4KiB", "1M", "1G" (binary units, case-insensitive). Throws std::invalid_argument.
uint64_t parse_size(const std::string& text);

// 4096 -> "4 KiB", 5888 -> "5.75 KiB".
std::string format_size(uint64_t bytes);

}  // namespace gpumem

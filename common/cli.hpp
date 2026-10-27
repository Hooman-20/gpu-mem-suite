#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "backend.hpp"
#include "sweep.hpp"

namespace gpumem {

struct Options {
  std::vector<Test> tests = {Test::Read, Test::Write, Test::Copy, Test::Chase};

  // Size sweep
  uint64_t min_bytes = 4 * KiB;
  uint64_t max_bytes = 1 * GiB;
  int points_per_octave = 2;

  // Timing
  int warmup = 3;
  int reps = 20;
  double target_ms = 1.0;  // minimum duration of one timed repetition (K launches)
  int max_k = 10000;

  // Correctness
  bool check_every_rep = false;
  uint32_t seed = 0x5eed1234U;

  // Pointer chase
  uint32_t chase_stride = 128;          // bytes between chase nodes
  uint64_t chase_steps = 0;             // 0 = auto (one lap, clamped)
  uint64_t chase_min_steps = 1ULL << 16;
  uint64_t chase_max_steps = 1ULL << 18;

  // Launch shape for streaming kernels (threads per block / threadgroup, blocks per SM / core)
  int block_size = 256;
  int blocks_per_cu = 4;

  int device = 0;
  std::string out_dir = "results";
};

struct ParseResult {
  Options options;
  bool show_help = false;
};

// Parses argv[1..] ("--flag value" or "--flag=value"). Throws std::invalid_argument with a
// user-facing message on bad input.
ParseResult parse_args(const std::vector<std::string>& args);

// Throws std::invalid_argument if `opts` is inconsistent.
void validate(const Options& opts);

std::string usage();

}  // namespace gpumem

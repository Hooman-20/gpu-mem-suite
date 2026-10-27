#include "cli.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace gpumem {
namespace {

long long parse_integer(const std::string& flag, const std::string& text) {
  size_t used = 0;
  long long value = 0;
  try {
    value = std::stoll(text, &used, 0);  // base 0 accepts 0x... for seeds
  } catch (const std::exception&) {
    throw std::invalid_argument(flag + ": expected an integer, got '" + text + "'");
  }
  if (used != text.size()) throw std::invalid_argument(flag + ": expected an integer, got '" + text + "'");
  return value;
}

int parse_int(const std::string& flag, const std::string& text) {
  const long long v = parse_integer(flag, text);
  if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(flag + ": value out of range");
  }
  return static_cast<int>(v);
}

double parse_double(const std::string& flag, const std::string& text) {
  size_t used = 0;
  double value = 0;
  try {
    value = std::stod(text, &used);
  } catch (const std::exception&) {
    throw std::invalid_argument(flag + ": expected a number, got '" + text + "'");
  }
  if (used != text.size()) throw std::invalid_argument(flag + ": expected a number, got '" + text + "'");
  return value;
}

uint64_t parse_count(const std::string& flag, const std::string& text) {
  const long long v = parse_integer(flag, text);
  if (v < 0) throw std::invalid_argument(flag + ": must be non-negative");
  return static_cast<uint64_t>(v);
}

}  // namespace

ParseResult parse_args(const std::vector<std::string>& args) {
  ParseResult result;
  Options& o = result.options;

  for (size_t i = 0; i < args.size(); ++i) {
    std::string flag = args[i];
    if (flag == "-h" || flag == "--help") {
      result.show_help = true;
      continue;
    }
    if (flag.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument '" + flag + "'");

    std::string inline_value;
    bool has_inline = false;
    if (const size_t eq = flag.find('='); eq != std::string::npos) {
      inline_value = flag.substr(eq + 1);
      flag = flag.substr(0, eq);
      has_inline = true;
    }
    auto value = [&]() -> std::string {
      if (has_inline) return inline_value;
      if (i + 1 >= args.size()) throw std::invalid_argument(flag + " needs a value");
      return args[++i];
    };

    if (flag == "--test") o.tests = parse_test_list(value());
    else if (flag == "--min-size") o.min_bytes = parse_size(value());
    else if (flag == "--max-size") o.max_bytes = parse_size(value());
    else if (flag == "--points-per-octave") o.points_per_octave = parse_int(flag, value());
    else if (flag == "--warmup") o.warmup = parse_int(flag, value());
    else if (flag == "--reps") o.reps = parse_int(flag, value());
    else if (flag == "--target-ms") o.target_ms = parse_double(flag, value());
    else if (flag == "--max-k") o.max_k = parse_int(flag, value());
    else if (flag == "--check-every-rep") {
      if (has_inline) throw std::invalid_argument("--check-every-rep takes no value");
      o.check_every_rep = true;
    }
    else if (flag == "--seed") {
      const uint64_t seed = parse_count(flag, value());
      if (seed > std::numeric_limits<uint32_t>::max()) throw std::invalid_argument("--seed must fit in 32 bits");
      o.seed = static_cast<uint32_t>(seed);
    }
    else if (flag == "--chase-stride") {
      const uint64_t stride = parse_size(value());
      if (stride > std::numeric_limits<uint32_t>::max()) throw std::invalid_argument("--chase-stride too large");
      o.chase_stride = static_cast<uint32_t>(stride);
    }
    else if (flag == "--chase-steps") o.chase_steps = parse_count(flag, value());
    else if (flag == "--chase-min-steps") o.chase_min_steps = parse_count(flag, value());
    else if (flag == "--chase-max-steps") o.chase_max_steps = parse_count(flag, value());
    else if (flag == "--block") o.block_size = parse_int(flag, value());
    else if (flag == "--blocks-per-cu") o.blocks_per_cu = parse_int(flag, value());
    else if (flag == "--device") o.device = parse_int(flag, value());
    else if (flag == "--out-dir") o.out_dir = value();
    else throw std::invalid_argument("unknown option '" + flag + "' (see --help)");
  }

  if (!result.show_help) validate(o);
  return result;
}

void validate(const Options& o) {
  auto require = [](bool ok, const std::string& msg) {
    if (!ok) throw std::invalid_argument(msg);
  };
  require(!o.tests.empty(), "no tests selected");
  require(o.min_bytes >= 256, "--min-size must be at least 256 bytes");
  require(o.min_bytes <= o.max_bytes, "--min-size must not exceed --max-size");
  require(o.max_bytes <= 16 * GiB, "--max-size above 16 GiB is not supported (32-bit word indices)");
  require(o.points_per_octave >= 1 && o.points_per_octave <= 16, "--points-per-octave must be in [1, 16]");
  require(o.warmup >= 0, "--warmup must be >= 0");
  require(o.reps >= 1, "--reps must be >= 1");
  require(o.target_ms > 0.0, "--target-ms must be > 0");
  require(o.max_k >= 1, "--max-k must be >= 1");
  require(o.block_size >= 32 && o.block_size <= 1024 && o.block_size % 32 == 0,
          "--block must be a multiple of 32 in [32, 1024]");
  require(o.blocks_per_cu >= 1, "--blocks-per-cu must be >= 1");
  require(o.device >= 0, "--device must be >= 0");
  require(!o.out_dir.empty(), "--out-dir must not be empty");

  const bool chase = std::find(o.tests.begin(), o.tests.end(), Test::Chase) != o.tests.end();
  if (chase) {
    require(o.chase_stride >= 4 && o.chase_stride % 4 == 0, "--chase-stride must be a positive multiple of 4 bytes");
    require(o.chase_stride <= o.min_bytes, "--chase-stride must not exceed --min-size");
    require(o.chase_min_steps >= 1, "--chase-min-steps must be >= 1");
    require(o.chase_min_steps <= o.chase_max_steps, "--chase-min-steps must not exceed --chase-max-steps");
  }
}

std::string usage() {
  return R"(gpumem - GPU memory bandwidth and latency characterization

Usage: gpumem [options]

Tests
  --test LIST            read|write|copy|chase|all, or a comma list (default: all)

Size sweep (sizes accept B, K/KB/KiB, M, G; binary units)
  --min-size SIZE        smallest buffer (default 4K)
  --max-size SIZE        largest buffer (default 1G)
  --points-per-octave N  sizes per doubling (default 2)

Timing
  --warmup N             untimed warm-up launches per size (default 3)
  --reps N               timed repetitions per size; median is reported (default 20)
  --target-ms MS         launches per rep (K) are chosen so a rep lasts >= MS (default 1.0)
  --max-k N              upper bound on K (default 10000)

Correctness
  --check-every-rep      verify output after every timed rep, not just once per size
  --seed N               fill-pattern / permutation seed (default 0x5eed1234)

Pointer chase
  --chase-stride SIZE    bytes between chase nodes (default 128)
  --chase-steps N        dependent loads per launch (default: auto = one lap)
  --chase-min-steps N    auto lower bound (default 65536)
  --chase-max-steps N    auto upper bound (default 262144)

Launch shape (streaming kernels)
  --block N              threads per block (default 256)
  --blocks-per-cu N      blocks per SM; grid = min(ceil(words/block), SMs * N) (default 4)

Other
  --device N             GPU index (default 0)
  --out-dir DIR          CSV directory (default results)
  -h, --help             show this help

Exit status: 0 = all checks passed, 2 = at least one size failed its check, 1 = error.
)";
}

}  // namespace gpumem

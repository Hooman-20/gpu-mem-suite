#include "cli.hpp"
#include "doctest.h"

#include <stdexcept>

using namespace gpumem;

TEST_CASE("defaults match the documented sweep and timing") {
  const ParseResult r = parse_args({});
  CHECK_FALSE(r.show_help);
  const Options& o = r.options;
  CHECK(o.tests.size() == 4);
  CHECK(o.min_bytes == 4 * KiB);
  CHECK(o.max_bytes == 1 * GiB);
  CHECK(o.warmup == 3);
  CHECK(o.reps == 20);
  CHECK(o.target_ms == 1.0);
  CHECK_FALSE(o.check_every_rep);
  CHECK(o.timing == TimingMode::Launch);
  CHECK(o.out_dir == "results");
}

TEST_CASE("--timing selects the launch mode") {
  CHECK(parse_args({"--timing", "graph"}).options.timing == TimingMode::Graph);
  CHECK(parse_args({"--timing=launch"}).options.timing == TimingMode::Launch);
  CHECK_THROWS_AS(parse_args({"--timing", "fast"}), std::invalid_argument);
}

TEST_CASE("--test selects tests") {
  CHECK(parse_args({"--test", "copy"}).options.tests == std::vector<Test>{Test::Copy});
  CHECK(parse_args({"--test=chase"}).options.tests == std::vector<Test>{Test::Chase});
  CHECK(parse_args({"--test", "read,chase,read"}).options.tests == std::vector<Test>{Test::Read, Test::Chase});
  CHECK(parse_args({"--test", "all"}).options.tests.size() == 4);
  CHECK_THROWS_AS(parse_args({"--test", "bogus"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_args({"--test"}), std::invalid_argument);
}

TEST_CASE("value flags parse in both spellings") {
  const Options o = parse_args({"--min-size", "64K", "--max-size=256M", "--reps", "5", "--warmup=0",
                                "--target-ms", "2.5", "--seed", "0x10", "--check-every-rep", "--out-dir", "/tmp/x",
                                "--chase-stride", "64", "--block", "512", "--blocks-per-cu", "8"})
                        .options;
  CHECK(o.min_bytes == 64 * KiB);
  CHECK(o.max_bytes == 256 * MiB);
  CHECK(o.reps == 5);
  CHECK(o.warmup == 0);
  CHECK(o.target_ms == 2.5);
  CHECK(o.seed == 16);
  CHECK(o.check_every_rep);
  CHECK(o.out_dir == "/tmp/x");
  CHECK(o.chase_stride == 64);
  CHECK(o.block_size == 512);
  CHECK(o.blocks_per_cu == 8);
}

TEST_CASE("help short-circuits validation") {
  CHECK(parse_args({"--help"}).show_help);
  CHECK(parse_args({"-h", "--reps", "0"}).show_help);
}

TEST_CASE("invalid options are rejected") {
  CHECK_THROWS_AS(parse_args({"--frobnicate"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_args({"stray"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_args({"--reps", "0"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_args({"--reps", "ten"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_args({"--min-size", "8K", "--max-size", "4K"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_args({"--min-size", "128"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_args({"--block", "100"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_args({"--seed", "0x100000000"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_args({"--check-every-rep=1"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_args({"--target-ms", "0"}), std::invalid_argument);
}

TEST_CASE("chase options are validated only when chase runs") {
  CHECK_THROWS_AS(parse_args({"--test", "chase", "--chase-stride", "6"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_args({"--test", "chase", "--min-size", "4K", "--chase-stride", "8K"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_args({"--test", "chase", "--chase-min-steps", "10", "--chase-max-steps", "5"}),
                  std::invalid_argument);
  CHECK_NOTHROW(parse_args({"--test", "read", "--chase-stride", "6"}));
}

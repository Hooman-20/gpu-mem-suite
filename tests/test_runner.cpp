// End-to-end runner tests against the CPU fake backend: sweep, K selection, checks, CSV output.

#include "doctest.h"
#include "fake_backend.hpp"
#include "runner.hpp"

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

using namespace gpumem;
using gpumem::testing::FakeBackend;
namespace fs = std::filesystem;

namespace {

struct ParsedCsv {
  std::map<std::string, std::string> meta;
  std::vector<std::string> header;
  std::vector<std::map<std::string, std::string>> rows;
};

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, ',')) out.push_back(field);
  if (!line.empty() && line.back() == ',') out.emplace_back();
  return out;
}

ParsedCsv read_csv(const std::string& path) {
  ParsedCsv csv;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("# ", 0) == 0) {
      const size_t colon = line.find(": ");
      csv.meta[line.substr(2, colon - 2)] = line.substr(colon + 2);
    } else if (csv.header.empty()) {
      csv.header = split(line);
    } else {
      const auto fields = split(line);
      REQUIRE(fields.size() == csv.header.size());
      std::map<std::string, std::string> row;
      for (size_t i = 0; i < fields.size(); ++i) row[csv.header[i]] = fields[i];
      csv.rows.push_back(row);
    }
  }
  return csv;
}

Options small_options(const std::string& out_dir) {
  Options o;
  o.min_bytes = 4 * KiB;
  o.max_bytes = 64 * KiB;
  o.points_per_octave = 1;
  o.warmup = 1;
  o.reps = 3;
  o.out_dir = out_dir;
  return o;
}

struct TempDir {
  fs::path path;
  explicit TempDir(const char* name) : path(fs::temp_directory_path() / name) {
    fs::remove_all(path);
  }
  ~TempDir() { fs::remove_all(path); }
};

}  // namespace

TEST_CASE("runner writes one passing CSV per test with device metadata") {
  TempDir dir("gpumem_runner_pass");
  FakeBackend backend;
  std::ostringstream log;
  const SuiteResult result = run_suite(backend, small_options(dir.path.string()), log);

  CHECK(result.failed_sizes == 0);
  REQUIRE(result.csv_paths.size() == 4);
  const char* names[] = {"read", "write", "copy", "chase"};
  for (size_t t = 0; t < 4; ++t) {
    const std::string& path = result.csv_paths[t];
    CAPTURE(path);
    CHECK(fs::path(path).filename().string().rfind(std::string("fake_") + names[t] + "_", 0) == 0);
    const ParsedCsv csv = read_csv(path);
    CHECK(csv.header == kCsvColumns);
    CHECK(csv.meta.at("device") == "Fake GPU, CPU-emulated");
    CHECK(csv.meta.at("l2_bytes") == "32768");
    CHECK(csv.meta.at("compute_units") == "2");
    CHECK(csv.meta.at("clock_khz") == "1000000");
    CHECK(csv.meta.at("peak_dram_gbs") == "123.4");
    CHECK(csv.meta.at("timing_mode") == "launch");
    CHECK(csv.meta.at("fake_extra") == "yes");
    CHECK(csv.meta.at("test") == names[t]);
    REQUIRE(csv.rows.size() == 5);  // 4K, 8K, 16K, 32K, 64K
    CHECK(csv.rows.front().at("size_bytes") == "4096");
    CHECK(csv.rows.back().at("size_bytes") == "65536");
    for (const auto& row : csv.rows) {
      CHECK(row.at("check") == "PASS");
      CHECK(row.at("mismatches") == "0");
      CHECK(row.at("checks") == "1");
      CHECK(row.at("reps") == "3");
      CHECK(row.at("k") == "3");  // ceil(1.0 ms / 0.4 ms)
      CHECK(row.at("median_ms") == "0.400000");
    }
  }
}

TEST_CASE("runner computes bandwidth from bytes moved and copy counts both directions") {
  TempDir dir("gpumem_runner_bw");
  FakeBackend backend;
  backend.ms_per_launch = 1.0;
  Options o = small_options(dir.path.string());
  o.tests = {Test::Read, Test::Copy};
  o.max_bytes = 4 * KiB;
  std::ostringstream log;
  const SuiteResult result = run_suite(backend, o, log);
  const auto read = read_csv(result.csv_paths[0]).rows.at(0);
  const auto copy = read_csv(result.csv_paths[1]).rows.at(0);
  CHECK(read.at("bytes_per_launch") == "4096");
  CHECK(read.at("bandwidth_gbs") == "0.00");  // 4096 B / 1 ms = 0.004 GB/s
  CHECK(copy.at("bytes_per_launch") == "8192");
  CHECK(read.at("cycles_per_load").empty());
}

TEST_CASE("chase rows report cycles and ns per load") {
  TempDir dir("gpumem_runner_chase");
  FakeBackend backend;
  Options o = small_options(dir.path.string());
  o.tests = {Test::Chase};
  o.chase_steps = 1000;
  std::ostringstream log;
  const auto rows = read_csv(run_suite(backend, o, log).csv_paths.at(0)).rows;
  for (const auto& row : rows) {
    CHECK(row.at("steps") == "1000");
    CHECK(row.at("cycles_per_load") == "10.00");  // fake reports 10 cycles per step
    CHECK(row.at("ns_per_load") == "400.000");    // 0.4 ms / 1000 loads
    CHECK(row.at("bandwidth_gbs").empty());
  }
}

TEST_CASE("broken kernels are flagged as FAIL with mismatch counts") {
  struct Case {
    FakeBackend::Fault fault;
    Test test;
    const char* mismatches;  // for the 4 KiB size
  };
  const Case cases[] = {
      {FakeBackend::Fault::DropLastWord, Test::Write, "1"},
      {FakeBackend::Fault::DropLastWord, Test::Copy, "1"},
      {FakeBackend::Fault::Nothing, Test::Copy, "1024"},
      {FakeBackend::Fault::SkipPartial, Test::Read, "1"},
      {FakeBackend::Fault::Nothing, Test::Read, "1"},
      {FakeBackend::Fault::WrongChase, Test::Chase, "1"},
      {FakeBackend::Fault::Nothing, Test::Chase, "1"},
  };
  for (const Case& c : cases) {
    CAPTURE(test_name(c.test));
    CAPTURE(static_cast<int>(c.fault));
    TempDir dir("gpumem_runner_fault");
    FakeBackend backend;
    backend.fault = c.fault;
    Options o = small_options(dir.path.string());
    o.tests = {c.test};
    std::ostringstream log;
    const SuiteResult result = run_suite(backend, o, log);
    CHECK(result.failed_sizes == 5);
    const auto rows = read_csv(result.csv_paths.at(0)).rows;
    CHECK(rows.at(0).at("check") == "FAIL");
    CHECK(rows.at(0).at("mismatches") == c.mismatches);
  }
}

TEST_CASE("--check-every-rep verifies after each timed repetition") {
  TempDir dir("gpumem_runner_every");
  FakeBackend backend;
  Options o = small_options(dir.path.string());
  o.check_every_rep = true;
  o.tests = {Test::Write, Test::Chase};
  std::ostringstream log;
  const SuiteResult result = run_suite(backend, o, log);
  CHECK(result.failed_sizes == 0);
  for (const auto& path : result.csv_paths) {
    const ParsedCsv csv = read_csv(path);
    CHECK(csv.meta.at("check_every_rep") == "1");
    for (const auto& row : csv.rows) CHECK(row.at("checks") == "3");
  }

  backend.fault = FakeBackend::Fault::DropLastWord;
  o.tests = {Test::Write};
  const auto rows = read_csv(run_suite(backend, o, log).csv_paths.at(0)).rows;
  CHECK(rows.at(0).at("check") == "FAIL");
  CHECK(rows.at(0).at("mismatches") == "3");  // one bad word in each of 3 reps
}

TEST_CASE("timing mode is recorded in the CSV header") {
  TempDir dir("gpumem_runner_timing");
  FakeBackend backend;
  Options o = small_options(dir.path.string());
  o.tests = {Test::Copy};
  o.max_bytes = 4 * KiB;
  o.timing = TimingMode::Graph;
  std::ostringstream log;
  CHECK(read_csv(run_suite(backend, o, log).csv_paths.at(0)).meta.at("timing_mode") == "graph");
}

TEST_CASE("K respects max_k and the per-launch time is reported") {
  TempDir dir("gpumem_runner_k");
  FakeBackend backend;
  backend.ms_per_launch = 0.001;
  Options o = small_options(dir.path.string());
  o.tests = {Test::Write};
  o.max_bytes = 4 * KiB;
  o.max_k = 50;
  std::ostringstream log;
  const auto row = read_csv(run_suite(backend, o, log).csv_paths.at(0)).rows.at(0);
  CHECK(row.at("k") == "50");
  CHECK(row.at("median_ms") == "0.001000");
  // warmup 1 + calibration (K=1 timed twice, then K jumps to max_k) + reps 3 * K 50 + check 1
  CHECK(backend.launches == 154);
}

TEST_CASE("K calibration reaches the target even when a single launch looks slow") {
  // Backend whose first launch of each size is 10x slower (launch jitter). Extrapolating from
  // that launch alone would pick K = 1; calibration must still make a rep last >= target.
  class JitteryBackend final : public Backend {
   public:
    FakeBackend inner;
    int calls = 0;
    DeviceInfo device_info() const override { return inner.device_info(); }
    DeviceBuffer alloc(size_t b) override { return inner.alloc(b); }
    void release(DeviceBuffer& b) override { inner.release(b); }
    void upload(DeviceBuffer d, const void* s, size_t b) override { inner.upload(d, s, b); }
    void download(void* d, DeviceBuffer s, size_t b) override { inner.download(d, s, b); }
    void zero(DeviceBuffer d, size_t b) override { inner.zero(d, b); }
    uint64_t read_partial_slots(uint64_t n) const override { return inner.read_partial_slots(n); }
    double time_launches(const KernelArgs& a, int k) override {
      inner.time_launches(a, k);
      return (++calls == 1 ? 1.0 : 0.0) + k * 0.1;  // first call: 1.1 ms for one launch
    }
  };
  TempDir dir("gpumem_runner_jitter");
  JitteryBackend backend;
  Options o = small_options(dir.path.string());
  o.tests = {Test::Write};
  o.max_bytes = 4 * KiB;
  o.warmup = 0;
  std::ostringstream log;
  const auto row = read_csv(run_suite(backend, o, log).csv_paths.at(0)).rows.at(0);
  CHECK(row.at("k") == "10");  // 10 * 0.1 ms = 1 ms target, despite the slow first launch
  CHECK(row.at("median_ms") == "0.100000");
}

#include "runner.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "chase.hpp"
#include "csv.hpp"
#include "hash.hpp"
#include "stats.hpp"
#include "sweep.hpp"

namespace gpumem {

const std::vector<std::string> kCsvColumns = {
    "test",          "size_bytes", "k",           "reps",            "median_ms",
    "min_ms",        "max_ms",     "bytes_per_launch", "bandwidth_gbs", "steps",
    "cycles_per_load", "ns_per_load", "checks",   "check",           "mismatches",
};

namespace {

class ScopedBuffer {
 public:
  ScopedBuffer(Backend& backend, size_t bytes) : backend_(backend), buf_(backend.alloc(bytes)) {}
  ~ScopedBuffer() { backend_.release(buf_); }
  ScopedBuffer(const ScopedBuffer&) = delete;
  ScopedBuffer& operator=(const ScopedBuffer&) = delete;
  DeviceBuffer get() const { return buf_; }

 private:
  Backend& backend_;
  DeviceBuffer buf_;
};

// One test's buffers plus its host-side expectation for the current size.
class Workload {
 public:
  explicit Workload(Backend& backend) : backend_(backend) {}
  virtual ~Workload() = default;

  virtual void prepare(uint64_t size_bytes) = 0;  // untimed: inputs on device, args filled
  virtual void clear_outputs() = 0;               // so a kernel that writes nothing cannot pass
  virtual uint64_t verify() = 0;                  // mismatching items for the last launch
  virtual uint64_t bytes_per_launch() const = 0;
  virtual uint64_t last_cycles() { return 0; }

  const KernelArgs& args() const { return args_; }

 protected:
  Backend& backend_;
  KernelArgs args_;
};

// read / write / copy over a prefix of a buffer filled with pattern_word(i, seed).
class StreamWorkload final : public Workload {
 public:
  StreamWorkload(Backend& backend, Test test, const std::vector<uint32_t>& pattern, uint32_t seed,
                 uint64_t max_slots)
      : Workload(backend), pattern_(pattern) {
    const size_t max_bytes = pattern.size() * sizeof(uint32_t);
    args_.test = test;
    args_.seed = seed;
    if (test == Test::Read || test == Test::Copy) {
      src_ = std::make_unique<ScopedBuffer>(backend, max_bytes);
      backend.upload(src_->get(), pattern.data(), max_bytes);
      args_.src = src_->get();
    }
    if (test == Test::Write || test == Test::Copy) {
      dst_ = std::make_unique<ScopedBuffer>(backend, max_bytes);
      args_.dst = dst_->get();
    }
    if (test == Test::Read) {
      partials_ = std::make_unique<ScopedBuffer>(backend, max_slots * sizeof(uint32_t));
      args_.partials = partials_->get();
    }
  }

  void prepare(uint64_t size_bytes) override {
    args_.n_words = size_bytes / sizeof(uint32_t);
    if (args_.test == Test::Read) {
      slots_ = backend_.read_partial_slots(args_.n_words);
      if (slots_ * sizeof(uint32_t) > args_.partials.bytes) throw std::logic_error("partials buffer too small");
      expected_sum_ = 0;
      for (uint64_t i = 0; i < args_.n_words; ++i) expected_sum_ += pattern_[i];
    }
  }

  void clear_outputs() override {
    if (args_.dst.ptr) backend_.zero(args_.dst, args_.n_words * sizeof(uint32_t));
    if (args_.partials.ptr) backend_.zero(args_.partials, slots_ * sizeof(uint32_t));
  }

  uint64_t verify() override {
    if (args_.test == Test::Read) {
      host_.resize(slots_);
      backend_.download(host_.data(), args_.partials, slots_ * sizeof(uint32_t));
      uint32_t sum = 0;
      for (uint32_t v : host_) sum += v;
      return sum == expected_sum_ ? 0 : 1;
    }
    host_.resize(args_.n_words);
    backend_.download(host_.data(), args_.dst, args_.n_words * sizeof(uint32_t));
    uint64_t bad = 0;
    for (uint64_t i = 0; i < args_.n_words; ++i) bad += host_[i] != pattern_[i];
    return bad;
  }

  uint64_t bytes_per_launch() const override {
    const uint64_t bytes = args_.n_words * sizeof(uint32_t);
    return args_.test == Test::Copy ? 2 * bytes : bytes;  // copy reads and writes every byte
  }

 private:
  const std::vector<uint32_t>& pattern_;
  std::unique_ptr<ScopedBuffer> src_, dst_, partials_;
  std::vector<uint32_t> host_;
  uint64_t slots_ = 0;
  uint32_t expected_sum_ = 0;  // wrapping uint32 sum, matching the kernel contract
};

class ChaseWorkload final : public Workload {
 public:
  ChaseWorkload(Backend& backend, uint64_t max_bytes, const Options& opts)
      : Workload(backend), buf_(backend, max_bytes), out_(backend, sizeof(ChaseResult)), opts_(opts) {
    args_.test = Test::Chase;
    args_.src = buf_.get();
    args_.chase_out = out_.get();
    args_.start = 0;
  }

  void prepare(uint64_t size_bytes) override {
    const uint64_t n_nodes = size_bytes / opts_.chase_stride;
    const auto next = make_cyclic_permutation(static_cast<uint32_t>(n_nodes),
                                              (static_cast<uint64_t>(opts_.seed) << 32) ^ size_bytes);
    const auto host = build_chase_buffer(next, opts_.chase_stride / sizeof(uint32_t));
    backend_.upload(args_.src, host.data(), host.size() * sizeof(uint32_t));
    args_.steps = choose_chase_steps(n_nodes, opts_.chase_steps, opts_.chase_min_steps, opts_.chase_max_steps);
    expected_final_ = predict_chase_final(host, args_.start, args_.steps);
  }

  void clear_outputs() override {
    const ChaseResult sentinel{0xffffffffU, 0, 0};
    backend_.upload(args_.chase_out, &sentinel, sizeof sentinel);
  }

  uint64_t verify() override { return read_result().final_index == expected_final_ ? 0 : 1; }
  uint64_t bytes_per_launch() const override { return 0; }
  uint64_t last_cycles() override { return read_result().cycles; }

 private:
  ChaseResult read_result() {
    ChaseResult r{};
    backend_.download(&r, args_.chase_out, sizeof r);
    return r;
  }

  ScopedBuffer buf_, out_;
  const Options& opts_;
  uint32_t expected_final_ = 0;
};

struct SizeResult {
  uint64_t size = 0;
  int k = 0;
  Summary ms;
  double median_cycles = 0.0;
  uint64_t bytes = 0;
  uint64_t steps = 0;
  int checks = 0;
  uint64_t mismatches = 0;
};

// Grows K until a batch of K launches lasts >= target_ms. Extrapolating from a single launch is
// unreliable: launch overhead jitters by several x (badly so under WSL/WDDM). Jitter only ever
// adds time, so each candidate batch is timed twice and the faster run is used.
int calibrate_k(Backend& backend, const KernelArgs& args, const Options& o) {
  int k = 1;
  for (int iter = 0; iter < 64 && k < o.max_k; ++iter) {
    const double ms = std::min(backend.time_launches(args, k), backend.time_launches(args, k));
    if (ms >= o.target_ms) break;
    k = std::max(choose_k(ms / k, o.target_ms, o.max_k), std::min(o.max_k, 2 * k));
  }
  return k;
}

SizeResult measure_size(Backend& backend, Workload& w, uint64_t size, const Options& o, bool cycles) {
  w.prepare(size);
  w.clear_outputs();
  for (int i = 0; i < o.warmup; ++i) backend.time_launches(w.args(), 1);

  SizeResult r;
  r.size = size;
  r.k = calibrate_k(backend, w.args(), o);
  r.bytes = w.bytes_per_launch();
  r.steps = w.args().steps;

  std::vector<double> rep_ms, rep_cycles;
  for (int rep = 0; rep < o.reps; ++rep) {
    if (o.check_every_rep) w.clear_outputs();
    rep_ms.push_back(backend.time_launches(w.args(), r.k) / r.k);
    if (cycles) rep_cycles.push_back(static_cast<double>(w.last_cycles()));
    if (o.check_every_rep) {
      r.mismatches += w.verify();
      ++r.checks;
    }
  }
  if (!o.check_every_rep) {
    w.clear_outputs();
    backend.time_launches(w.args(), 1);
    r.mismatches += w.verify();
    r.checks = 1;
  }

  r.ms = summarize(rep_ms);
  if (!rep_cycles.empty()) r.median_cycles = median(rep_cycles);
  return r;
}

std::vector<std::string> csv_row(Test test, const SizeResult& r, int reps, bool cycles) {
  const bool chase = test == Test::Chase;
  const double gbs = r.bytes / (r.ms.median * 1e-3) / 1e9;
  return {
      test_name(test),
      std::to_string(r.size),
      std::to_string(r.k),
      std::to_string(reps),
      format_fixed(r.ms.median, 6),
      format_fixed(r.ms.min, 6),
      format_fixed(r.ms.max, 6),
      chase ? "" : std::to_string(r.bytes),
      chase ? "" : format_fixed(gbs, 2),
      chase ? std::to_string(r.steps) : "",
      chase && cycles ? format_fixed(r.median_cycles / r.steps, 2) : "",
      chase ? format_fixed(r.ms.median * 1e6 / r.steps, 3) : "",
      std::to_string(r.checks),
      r.mismatches == 0 ? "PASS" : "FAIL",
      std::to_string(r.mismatches),
  };
}

void write_preamble(CsvWriter& csv, const DeviceInfo& info, const Options& o, Test test, const std::string& stamp) {
  char seed[16];
  std::snprintf(seed, sizeof seed, "0x%08x", o.seed);
  csv.comment("tool", "gpumem");
  csv.comment("backend", info.backend);
  csv.comment("device", info.name);
  csv.comment("compute_units", std::to_string(info.compute_units));
  csv.comment("l2_bytes", std::to_string(info.l2_bytes));
  csv.comment("clock_khz", std::to_string(info.clock_khz));
  csv.comment("global_mem_bytes", std::to_string(info.global_mem_bytes));
  csv.comment("has_cycle_counter", info.has_cycle_counter ? "1" : "0");
  for (const auto& [key, value] : info.extra) csv.comment(key, value);
  csv.comment("test", test_name(test));
  csv.comment("timestamp", stamp);
  csv.comment("warmup", std::to_string(o.warmup));
  csv.comment("reps", std::to_string(o.reps));
  csv.comment("target_ms", format_fixed(o.target_ms, 3));
  csv.comment("max_k", std::to_string(o.max_k));
  csv.comment("seed", seed);
  csv.comment("check_every_rep", o.check_every_rep ? "1" : "0");
  csv.comment("points_per_octave", std::to_string(o.points_per_octave));
  if (test == Test::Chase) csv.comment("chase_stride_bytes", std::to_string(o.chase_stride));
  csv.header(kCsvColumns);
}

void log_size(std::ostream& log, Test test, const SizeResult& r, bool cycles) {
  char line[160];
  if (test == Test::Chase) {
    std::snprintf(line, sizeof line, "  %-5s %12s  k=%-5d %8.2f cyc/load %8.2f ns/load  %s",
                  test_name(test), format_size(r.size).c_str(), r.k, cycles ? r.median_cycles / r.steps : 0.0,
                  r.ms.median * 1e6 / r.steps, r.mismatches ? "FAIL" : "PASS");
  } else {
    std::snprintf(line, sizeof line, "  %-5s %12s  k=%-5d %10.4f ms %9.1f GB/s  %s", test_name(test),
                  format_size(r.size).c_str(), r.k, r.ms.median, r.bytes / (r.ms.median * 1e-3) / 1e9,
                  r.mismatches ? "FAIL" : "PASS");
  }
  log << line << std::endl;
}

}  // namespace

SuiteResult run_suite(Backend& backend, const Options& opts, std::ostream& log) {
  validate(opts);
  const DeviceInfo info = backend.device_info();
  const bool cycles = info.has_cycle_counter;
  const auto sizes = make_size_sweep(opts.min_bytes, opts.max_bytes, opts.points_per_octave, kSizeAlign);
  const uint64_t max_bytes = sizes.back();
  const std::string stamp = timestamp_now();
  std::filesystem::create_directories(opts.out_dir);

  std::vector<uint32_t> pattern;  // shared by read/write/copy, generated on first use
  SuiteResult result;

  for (Test test : opts.tests) {
    std::unique_ptr<Workload> workload;
    if (test == Test::Chase) {
      workload = std::make_unique<ChaseWorkload>(backend, max_bytes, opts);
    } else {
      if (pattern.empty()) {
        pattern.resize(max_bytes / sizeof(uint32_t));
        for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = pattern_word(i, opts.seed);
      }
      uint64_t max_slots = 0;
      for (uint64_t s : sizes) max_slots = std::max(max_slots, backend.read_partial_slots(s / sizeof(uint32_t)));
      workload = std::make_unique<StreamWorkload>(backend, test, pattern, opts.seed, max_slots);
    }

    const std::string path = make_results_path(opts.out_dir, info.backend, test_name(test), stamp);
    std::ofstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path + " for writing");
    CsvWriter csv(file);
    write_preamble(csv, info, opts, test, stamp);

    log << test_name(test) << " -> " << path << std::endl;
    for (uint64_t size : sizes) {
      const SizeResult r = measure_size(backend, *workload, size, opts, cycles && test == Test::Chase);
      csv.row(csv_row(test, r, opts.reps, cycles));
      log_size(log, test, r, cycles);
      if (r.mismatches) ++result.failed_sizes;
    }
    result.csv_paths.push_back(path);
  }
  return result;
}

}  // namespace gpumem

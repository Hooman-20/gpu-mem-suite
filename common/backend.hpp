#pragma once
// Backend-agnostic interface. The runner only talks to a GPU through this; cuda/ implements it
// today and metal/ will implement it on macOS.

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "kernel_abi.hpp"

namespace gpumem {

enum class Test { Read, Write, Copy, Chase };

// How a backend issues the K launches of one timed repetition.
//   Launch: K individual launches, back to back (includes per-launch submission overhead).
//   Graph:  K launches pre-recorded and submitted as one unit (CUDA Graphs; for Metal, one
//           command buffer). Backends that cannot do this reject it at construction.
enum class TimingMode { Launch, Graph };

const char* test_name(Test test);

const char* timing_mode_name(TimingMode mode);
TimingMode parse_timing_mode(const std::string& text);  // "launch" | "graph"; throws std::invalid_argument

// "read", "write", "copy", "chase", "all", or a comma-separated list. Throws std::invalid_argument.
std::vector<Test> parse_test_list(const std::string& text);

// Nominal DRAM bandwidth in GB/s: 2 transfers per reported memory clock across the bus
// (the convention CUDA's memoryClockRate follows, GDDR7 included). 0 if either input is unknown.
double nominal_dram_gbs(uint64_t mem_clock_khz, int bus_width_bits);

// What a backend knows about its device. Anything the API cannot report stays 0.
struct DeviceInfo {
  std::string backend;             // "cuda", "metal", ...
  std::string name;
  int compute_units = 0;           // SMs on CUDA, GPU cores on Apple silicon
  uint64_t l2_bytes = 0;           // last-level on-chip cache size reported by the API
  uint64_t global_mem_bytes = 0;
  int clock_khz = 0;               // max shader clock reported by the API
  double peak_dram_gbs = 0.0;      // nominal DRAM bandwidth derived from the API (0 = unknown)
  bool has_cycle_counter = false;  // kernels can read a cycle counter (clock64() on CUDA)
  std::vector<std::pair<std::string, std::string>> extra;  // backend-specific CSV header fields
};

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;
};

// Everything one kernel launch needs; which fields matter depends on `test`.
struct KernelArgs {
  Test test = Test::Read;
  DeviceBuffer src;        // read, copy: input words. chase: the chase buffer
  DeviceBuffer dst;        // write, copy: output words
  DeviceBuffer partials;   // read: one uint32 per launched thread
  DeviceBuffer chase_out;  // chase: one ChaseResult
  uint64_t n_words = 0;    // read/write/copy: 32-bit words to touch
  uint32_t seed = 0;       // write: pattern seed
  uint32_t start = 0;      // chase: starting word index
  uint64_t steps = 0;      // chase: dependent loads per launch
};

class Backend {
 public:
  virtual ~Backend() = default;

  virtual DeviceInfo device_info() const = 0;

  virtual DeviceBuffer alloc(size_t bytes) = 0;
  virtual void release(DeviceBuffer& buf) = 0;
  virtual void upload(DeviceBuffer dst, const void* src, size_t bytes) = 0;
  virtual void download(void* dst, DeviceBuffer src, size_t bytes) = 0;
  virtual void zero(DeviceBuffer dst, size_t bytes) = 0;

  // uint32 partial sums the read kernel writes for a launch over n_words (= threads launched).
  virtual uint64_t read_partial_slots(uint64_t n_words) const = 0;

  // Launches the kernel for args.test k times back to back and returns GPU time for all k, in ms.
  virtual double time_launches(const KernelArgs& args, int k) = 0;
};

}  // namespace gpumem

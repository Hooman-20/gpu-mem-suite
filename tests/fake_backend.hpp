#pragma once
// CPU stand-in for a GPU backend so the runner can be tested without a GPU. "Kernels" follow
// the contracts in cuda/kernels.cu; faults let tests check that broken output is caught.

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "backend.hpp"
#include "hash.hpp"

namespace gpumem::testing {

class FakeBackend final : public Backend {
 public:
  enum class Fault { None, DropLastWord, WrongChase, SkipPartial, Nothing };

  Fault fault = Fault::None;
  double ms_per_launch = 0.4;  // fixed fake GPU time per launch
  uint64_t threads = 64;       // fake grid size for the read kernel
  int launches = 0;

  DeviceInfo device_info() const override {
    DeviceInfo d;
    d.backend = "fake";
    d.name = "Fake GPU, CPU-emulated";
    d.compute_units = 2;
    d.l2_bytes = 32 * 1024;
    d.global_mem_bytes = 1ULL << 30;
    d.clock_khz = 1000000;
    d.peak_dram_gbs = 123.4;
    d.has_cycle_counter = true;
    d.extra = {{"fake_extra", "yes"}};
    return d;
  }

  DeviceBuffer alloc(size_t bytes) override {
    DeviceBuffer b;
    b.ptr = std::calloc(bytes ? bytes : 1, 1);
    if (!b.ptr) throw std::bad_alloc();
    b.bytes = bytes;
    return b;
  }
  void release(DeviceBuffer& b) override {
    std::free(b.ptr);
    b = DeviceBuffer{};
  }
  void upload(DeviceBuffer dst, const void* src, size_t bytes) override {
    bounds(dst, bytes);
    std::memcpy(dst.ptr, src, bytes);
  }
  void download(void* dst, DeviceBuffer src, size_t bytes) override {
    bounds(src, bytes);
    std::memcpy(dst, src.ptr, bytes);
  }
  void zero(DeviceBuffer dst, size_t bytes) override {
    bounds(dst, bytes);
    std::memset(dst.ptr, 0, bytes);
  }

  uint64_t read_partial_slots(uint64_t n_words) const override { return n_words < threads ? n_words : threads; }

  double time_launches(const KernelArgs& a, int k) override {
    for (int i = 0; i < k; ++i) run(a);
    launches += k;
    return k * ms_per_launch;
  }

 private:
  static void bounds(DeviceBuffer b, size_t bytes) {
    if (bytes > b.bytes) throw std::out_of_range("fake backend: access past end of buffer");
  }

  void run(const KernelArgs& a) {
    if (fault == Fault::Nothing) return;
    const auto* src = static_cast<const uint32_t*>(a.src.ptr);
    auto* dst = static_cast<uint32_t*>(a.dst.ptr);
    const uint64_t n = a.n_words - (fault == Fault::DropLastWord ? 1 : 0);
    switch (a.test) {
      case Test::Read: {
        auto* partials = static_cast<uint32_t*>(a.partials.ptr);
        const uint64_t t_count = read_partial_slots(a.n_words);
        for (uint64_t t = 0; t < t_count; ++t) {
          uint32_t sum = 0;
          for (uint64_t i = t; i < n; i += t_count) sum += src[i];
          if (!(fault == Fault::SkipPartial && t == 0)) partials[t] = sum;
        }
        break;
      }
      case Test::Write:
        for (uint64_t i = 0; i < n; ++i) dst[i] = pattern_word(i, a.seed);
        break;
      case Test::Copy:
        for (uint64_t i = 0; i < n; ++i) dst[i] = src[i];
        break;
      case Test::Chase: {
        uint32_t idx = a.start;
        for (uint64_t s = 0; s < a.steps; ++s) idx = src[idx];
        auto* out = static_cast<ChaseResult*>(a.chase_out.ptr);
        out->final_index = fault == Fault::WrongChase ? idx + 1 : idx;
        out->cycles = a.steps * 10;
        break;
      }
    }
  }
};

}  // namespace gpumem::testing

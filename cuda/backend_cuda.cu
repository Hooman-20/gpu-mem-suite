#include "backend_cuda.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace gpumem {
namespace {

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

int attribute(cudaDeviceAttr attr, int device) {
  int value = 0;
  check(cudaDeviceGetAttribute(&value, attr, device), "cudaDeviceGetAttribute");
  return value;
}

}  // namespace

CudaBackend::CudaBackend(int device, int block_size, int blocks_per_sm)
    : device_(device), block_size_(block_size), blocks_per_sm_(blocks_per_sm) {
  int count = 0;
  check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (device < 0 || device >= count) {
    throw std::runtime_error("CUDA device " + std::to_string(device) + " not found (" + std::to_string(count) +
                             " visible)");
  }
  check(cudaSetDevice(device), "cudaSetDevice");
  sm_count_ = attribute(cudaDevAttrMultiProcessorCount, device);
  check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
  check(cudaEventCreate(&start_), "cudaEventCreate");
  check(cudaEventCreate(&stop_), "cudaEventCreate");
}

CudaBackend::~CudaBackend() {
  if (start_) cudaEventDestroy(start_);
  if (stop_) cudaEventDestroy(stop_);
  if (stream_) cudaStreamDestroy(stream_);
}

DeviceInfo CudaBackend::device_info() const {
  cudaDeviceProp prop{};
  check(cudaGetDeviceProperties(&prop, device_), "cudaGetDeviceProperties");
  int runtime = 0, driver = 0;
  check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
  check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");

  DeviceInfo info;
  info.backend = "cuda";
  info.name = prop.name;
  info.compute_units = sm_count_;
  info.l2_bytes = static_cast<uint64_t>(attribute(cudaDevAttrL2CacheSize, device_));
  info.global_mem_bytes = prop.totalGlobalMem;
  info.clock_khz = attribute(cudaDevAttrClockRate, device_);
  info.has_cycle_counter = true;
  info.extra = {
      {"compute_capability", std::to_string(prop.major) + "." + std::to_string(prop.minor)},
      {"mem_clock_khz", std::to_string(attribute(cudaDevAttrMemoryClockRate, device_))},
      {"mem_bus_width_bits", std::to_string(attribute(cudaDevAttrGlobalMemoryBusWidth, device_))},
      {"shared_mem_per_sm_bytes", std::to_string(attribute(cudaDevAttrMaxSharedMemoryPerMultiprocessor, device_))},
      {"max_persisting_l2_bytes", std::to_string(attribute(cudaDevAttrMaxPersistingL2CacheSize, device_))},
      {"cuda_runtime_version", std::to_string(runtime)},
      {"cuda_driver_version", std::to_string(driver)},
      {"launch_block_size", std::to_string(block_size_)},
      {"launch_blocks_per_sm", std::to_string(blocks_per_sm_)},
  };
  return info;
}

DeviceBuffer CudaBackend::alloc(size_t bytes) {
  DeviceBuffer buf;
  if (bytes == 0) return buf;
  check(cudaMalloc(&buf.ptr, bytes), "cudaMalloc");
  buf.bytes = bytes;
  return buf;
}

void CudaBackend::release(DeviceBuffer& buf) {
  if (buf.ptr) cudaFree(buf.ptr);
  buf = DeviceBuffer{};
}

void CudaBackend::upload(DeviceBuffer dst, const void* src, size_t bytes) {
  check(cudaMemcpyAsync(dst.ptr, src, bytes, cudaMemcpyHostToDevice, stream_), "cudaMemcpy H2D");
  check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

void CudaBackend::download(void* dst, DeviceBuffer src, size_t bytes) {
  check(cudaMemcpyAsync(dst, src.ptr, bytes, cudaMemcpyDeviceToHost, stream_), "cudaMemcpy D2H");
  check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

void CudaBackend::zero(DeviceBuffer dst, size_t bytes) {
  check(cudaMemsetAsync(dst.ptr, 0, bytes, stream_), "cudaMemset");
  check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

LaunchConfig CudaBackend::stream_config(uint64_t n_words) const {
  const uint64_t needed = (n_words + block_size_ - 1) / block_size_;
  const uint64_t cap = static_cast<uint64_t>(sm_count_) * blocks_per_sm_;
  const auto blocks = static_cast<unsigned>(std::max<uint64_t>(1, std::min(needed, cap)));
  return {dim3(blocks), dim3(static_cast<unsigned>(block_size_))};
}

uint64_t CudaBackend::read_partial_slots(uint64_t n_words) const {
  const LaunchConfig cfg = stream_config(n_words);
  return static_cast<uint64_t>(cfg.grid.x) * cfg.block.x;
}

void CudaBackend::launch(const KernelArgs& a) {
  switch (a.test) {
    case Test::Read:
      launch_read_bw(stream_config(a.n_words), stream_, static_cast<const uint32_t*>(a.src.ptr), a.n_words,
                     static_cast<uint32_t*>(a.partials.ptr));
      break;
    case Test::Write:
      launch_write_bw(stream_config(a.n_words), stream_, static_cast<uint32_t*>(a.dst.ptr), a.n_words, a.seed);
      break;
    case Test::Copy:
      launch_copy_bw(stream_config(a.n_words), stream_, static_cast<const uint32_t*>(a.src.ptr),
                     static_cast<uint32_t*>(a.dst.ptr), a.n_words);
      break;
    case Test::Chase:
      launch_pointer_chase(stream_, static_cast<const uint32_t*>(a.src.ptr), a.start, a.steps,
                           static_cast<ChaseResult*>(a.chase_out.ptr));
      break;
  }
}

double CudaBackend::time_launches(const KernelArgs& args, int k) {
  check(cudaEventRecord(start_, stream_), "cudaEventRecord");
  for (int i = 0; i < k; ++i) launch(args);
  check(cudaGetLastError(), "kernel launch");
  check(cudaEventRecord(stop_, stream_), "cudaEventRecord");
  check(cudaEventSynchronize(stop_), "kernel execution");
  float ms = 0.0f;
  check(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
  return ms;
}

}  // namespace gpumem

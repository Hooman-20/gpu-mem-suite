#pragma once

#include <cuda_runtime.h>

#include "backend.hpp"
#include "kernels.cuh"

namespace gpumem {

class CudaBackend final : public Backend {
 public:
  CudaBackend(int device, int block_size, int blocks_per_sm);
  ~CudaBackend() override;
  CudaBackend(const CudaBackend&) = delete;
  CudaBackend& operator=(const CudaBackend&) = delete;

  DeviceInfo device_info() const override;
  DeviceBuffer alloc(size_t bytes) override;
  void release(DeviceBuffer& buf) override;
  void upload(DeviceBuffer dst, const void* src, size_t bytes) override;
  void download(void* dst, DeviceBuffer src, size_t bytes) override;
  void zero(DeviceBuffer dst, size_t bytes) override;
  uint64_t read_partial_slots(uint64_t n_words) const override;
  double time_launches(const KernelArgs& args, int k) override;

 private:
  LaunchConfig stream_config(uint64_t n_words) const;
  void launch(const KernelArgs& args);

  int device_;
  int block_size_;
  int blocks_per_sm_;
  int sm_count_ = 0;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

}  // namespace gpumem

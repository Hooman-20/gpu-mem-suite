#include <memory>

#include "backend_cuda.hpp"
#include "gpumem_main.hpp"

int main(int argc, char** argv) {
  return gpumem::gpumem_main(argc, argv, [](const gpumem::Options& opts) -> std::unique_ptr<gpumem::Backend> {
    return std::make_unique<gpumem::CudaBackend>(opts.device, opts.block_size, opts.blocks_per_cu, opts.timing);
  });
}

#include "gpumem_main.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "runner.hpp"
#include "sweep.hpp"

namespace gpumem {

int gpumem_main(int argc, char** argv, const BackendFactory& make_backend) {
  try {
    const ParseResult parsed = parse_args(std::vector<std::string>(argv + 1, argv + argc));
    if (parsed.show_help) {
      std::cout << usage();
      return 0;
    }
    const Options& opts = parsed.options;
    const std::unique_ptr<Backend> backend = make_backend(opts);
    const DeviceInfo info = backend->device_info();
    std::cout << "Device: " << info.name << " [" << info.backend << "]  units=" << info.compute_units
              << "  L2=" << format_size(info.l2_bytes) << "  clock=" << info.clock_khz / 1000 << " MHz"
              << "  DRAM peak=" << static_cast<int>(info.peak_dram_gbs) << " GB/s\n";

    const SuiteResult result = run_suite(*backend, opts, std::cout);
    if (result.failed_sizes > 0) {
      std::cerr << "FAILED: " << result.failed_sizes << " size(s) did not pass the correctness check\n";
      return 2;
    }
    std::cout << "All checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "gpumem: " << e.what() << '\n';
    return 1;
  }
}

}  // namespace gpumem

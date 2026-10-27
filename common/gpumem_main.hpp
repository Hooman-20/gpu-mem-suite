#pragma once
// Shared entry point. Each backend's main() only supplies a factory for its Backend.

#include <functional>
#include <memory>

#include "backend.hpp"
#include "cli.hpp"

namespace gpumem {

using BackendFactory = std::function<std::unique_ptr<Backend>(const Options&)>;

int gpumem_main(int argc, char** argv, const BackendFactory& make_backend);

}  // namespace gpumem

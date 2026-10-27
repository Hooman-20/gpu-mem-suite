# gpumem build. `make test` only needs a host C++17 compiler (no CUDA, no GPU).

NVCC ?= nvcc
CXX  ?= g++
ARCH ?= sm_120

BUILD    := build
BIN      := bin/gpumem
TEST_BIN := $(BUILD)/unit_tests

CPPFLAGS  := -Icommon
CXXFLAGS  := -O3 -std=c++17 -Wall -Wextra -Wpedantic
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -Wall,-Wextra

COMMON_SRCS := $(wildcard common/*.cpp)
CUDA_SRCS   := $(wildcard cuda/*.cu)
TEST_SRCS   := $(wildcard tests/*.cpp)

COMMON_OBJS := $(COMMON_SRCS:%.cpp=$(BUILD)/%.o)
CUDA_OBJS   := $(CUDA_SRCS:%.cu=$(BUILD)/%.o)
TEST_OBJS   := $(TEST_SRCS:%.cpp=$(BUILD)/%.o)

# Extra arguments for `make run`, e.g. make run ARGS="--test read --max-size 256M"
ARGS ?= --test all
SMOKE_ARGS := --test all --min-size 4K --max-size 1M --points-per-octave 1 --warmup 1 --reps 3 \
              --out-dir results/smoke

.PHONY: all run test smoke clean

all: $(BIN)

$(BIN): $(COMMON_OBJS) $(CUDA_OBJS)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD)/common/%.o: common/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/cuda/%.o: cuda/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(CPPFLAGS) -Icuda $(NVCCFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/tests/%.o: tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -Itests -Itests/third_party $(CXXFLAGS) -MMD -MP -c $< -o $@

$(TEST_BIN): $(COMMON_OBJS) $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(BIN)
	./$(BIN) $(ARGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

smoke: $(BIN)
	./$(BIN) $(SMOKE_ARGS)

clean:
	rm -rf $(BUILD) bin

-include $(COMMON_OBJS:.o=.d) $(CUDA_OBJS:.o=.d) $(TEST_OBJS:.o=.d)

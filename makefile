GCC_PATH := /apps/gcc-9.1.0/local

# Specific OneDNN path for OpenMP (Shared Threading)
# We go 'up' one level from the current DNNLROOT to find the sibling folder 'cpu_iomp'
# This is done to avoid building with conflicting versions of omp (between OneDNN and our code)
DNNL_IOMP_ROOT := $(shell dirname $(DNNLROOT))/cpu_iomp

CXX := dpcpp

# Compile Flags
CXXFLAGS := -std=c++17 -mavx512f -mavx512vnni -mfma -qopenmp -Ofast \
            -I$(DNNLROOT)/include -Iinclude \
            --gcc-toolchain=$(GCC_PATH)

# Linker Flags - THE FIX IS HERE
# 1. We link against $(DNNL_IOMP_ROOT)/lib instead of $(DNNLROOT)/lib
# 2. We add rpaths for both the GCC libs and this specific OneDNN lib
LDFLAGS  := -L$(DNNL_IOMP_ROOT)/lib -ldnnl -liomp5 \
            -Wl,-rpath,$(GCC_PATH)/lib64 \
            -Wl,-rpath,$(DNNL_IOMP_ROOT)/lib

SRC_DIR   := src
BUILD_DIR := build

ALL_SRCS  := $(wildcard $(SRC_DIR)/*.cpp)
MAIN_SRCS   := $(filter $(SRC_DIR)/main_%.cpp, $(ALL_SRCS))
COMMON_SRCS := $(filter-out $(SRC_DIR)/main_%.cpp, $(ALL_SRCS))
COMMON_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(COMMON_SRCS))
EXECUTABLES := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%,$(MAIN_SRCS))

all: $(BUILD_DIR) $(EXECUTABLES)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXECUTABLES): $(BUILD_DIR)/%: $(SRC_DIR)/%.cpp $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $< $(COMMON_OBJS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
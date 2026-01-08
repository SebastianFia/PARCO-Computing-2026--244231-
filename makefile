# Compiler settings
CXX := dpcpp
CXXFLAGS := -std=c++17 -O3 -I$(DNNLROOT)/include -Iinclude
LDFLAGS := -L$(DNNLROOT)/lib -ldnnl -lpthread

# Directories
SRC_DIR := src
BUILD_DIR := build

# Find all main files (e.g., src/main_test.cpp, src/main_bench.cpp)
MAIN_SRCS := $(wildcard $(SRC_DIR)/main_*.cpp)
# Generate corresponding executable names in build/
EXECUTABLES := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%,$(MAIN_SRCS))

# Shared object files (gemm_onednn implementation)
COMMON_SRCS := $(SRC_DIR)/gemm_onednn.cpp
COMMON_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(COMMON_SRCS))

# Default target
all: $(BUILD_DIR) $(EXECUTABLES)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule to build the common implementation object
$(BUILD_DIR)/gemm_onednn.o: $(SRC_DIR)/gemm_onednn.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to build each main executable
# Links the specific main object with the common gemm_onednn object
$(BUILD_DIR)/%: $(SRC_DIR)/%.cpp $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $< $(COMMON_OBJS) -o $@ $(LDFLAGS)

# Clean target
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
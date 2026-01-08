# Compiler and Flags
CXX := dpcpp
CXXFLAGS := -std=c++17 -O3 -I$(DNNLROOT)/include -Iinclude
LDFLAGS  := -L$(DNNLROOT)/lib -ldnnl -lpthread

# Directories
SRC_DIR   := src
BUILD_DIR := build

# 1. Find all .cpp files in src/
ALL_SRCS := $(wildcard $(SRC_DIR)/*.cpp)

# 2. Separate "Mains" from "Common" files
# Any file starting with "src/main_" is treated as an executable entry point.
MAIN_SRCS := $(filter $(SRC_DIR)/main_%.cpp, $(ALL_SRCS))

# Any file NOT starting with "src/main_" is treated as common code to be compiled and linked.
COMMON_SRCS := $(filter-out $(SRC_DIR)/main_%.cpp, $(ALL_SRCS))

# 3. Define Output Artifacts
# Convert common .cpp files to .o files in the build folder
COMMON_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(COMMON_SRCS))

# Convert main .cpp files to executables in the build folder (strip extension)
EXECUTABLES := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%,$(MAIN_SRCS))

# --- Targets ---

all: $(BUILD_DIR) $(EXECUTABLES)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule to compile Common Objects (e.g., gemm_onednn.cpp -> build/gemm_onednn.o)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile and link Executables
# Uses a Static Pattern Rule: targets: target-pattern: prereq-patterns
# This says: "For every executable, it depends on its specific .cpp file AND ALL common objects"
$(EXECUTABLES): $(BUILD_DIR)/%: $(SRC_DIR)/%.cpp $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $< $(COMMON_OBJS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
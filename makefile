# --- CONFIGURATION ---

# The path derived from your 'module show' output
GCC_PATH := /apps/gcc-9.1.0/local

# Compiler
CXX := dpcpp

# Flags:
# 1. --gcc-toolchain points dpcpp to the GCC 9.1 headers/libs explicitly.
# 2. -Isrc ensures we can find headers like "matrix.hpp" inside src/
CXXFLAGS := -std=c++17 -mavx2 -mfma -fopenmp -O3 \
            -I$(DNNLROOT)/include -Iinclude \
            --gcc-toolchain=$(GCC_PATH)

# Linker Flags:
# 1. -Wl,-rpath adds the GCC 9.1 lib folder to the runtime search path
#    so you don't get "version `GLIBCXX_3.4.xx' not found" errors when running.
LDFLAGS  := -L$(DNNLROOT)/lib -ldnnl -lpthread \
            -Wl,-rpath,$(GCC_PATH)/lib64

# --- DIRECTORIES ---

SRC_DIR   := src
BUILD_DIR := build

# --- SOURCE DISCOVERY ---

# 1. Find all .cpp files in src/
ALL_SRCS := $(wildcard $(SRC_DIR)/*.cpp)

# 2. Separate "Mains" (entry points) from "Common" (implementation) files
MAIN_SRCS   := $(filter $(SRC_DIR)/main_%.cpp, $(ALL_SRCS))
COMMON_SRCS := $(filter-out $(SRC_DIR)/main_%.cpp, $(ALL_SRCS))

# 3. Define Output Artifacts
COMMON_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(COMMON_SRCS))
EXECUTABLES := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%,$(MAIN_SRCS))

# --- TARGETS ---

all: $(BUILD_DIR) $(EXECUTABLES)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule to compile Common Objects
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile and link Executables
# Depends on the specific main file AND all common objects
$(EXECUTABLES): $(BUILD_DIR)/%: $(SRC_DIR)/%.cpp $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $< $(COMMON_OBJS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
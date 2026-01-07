# --- Configuration ---
CXX          := dpcpp
ONEDNN_ROOT  := ./oneDNN

# Compiler Flags
# -MMD -MP: Generates dependency files (.d) automatically
CXXFLAGS     := -std=c++17 -Ofast -march=native -mavx2 -mfma -fopenmp
CXXFLAGS     += -MMD -MP
CXXFLAGS     += -Iinclude -I$(ONEDNN_ROOT)/include -I$(ONEDNN_ROOT)/build/include

# Linker Flags
LDFLAGS      := -L$(ONEDNN_ROOT)/build/src
LDFLAGS      += -Wl,-rpath,$(ONEDNN_ROOT)/build/src
LDLIBS       := -ldnnl

# --- Auto-Discovery ---
# 1. Find sources: src/main_xyz.cpp
SRCS := $(wildcard src/main_*.cpp)

# 2. Define targets: src/main_xyz.cpp -> build/xyz
BINS := $(patsubst src/main_%.cpp,build/%,$(SRCS))

# --- Targets ---
.PHONY: all clean info

all: $(BINS)

# Build rule
# The dependency file (build/xyz.d) is generated automatically by the compiler
# due to the -MMD flag.
build/%: src/main_%.cpp
	@mkdir -p build
	@echo "Building $@ from $<..."
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf build

# --- Dependency Inclusion ---
# This line pulls in the generated dependency files.
# The '-' at the start suppresses errors if the .d files don't exist yet.
-include $(BINS:=.d)
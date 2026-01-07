# --- Configuration ---
CXX          := dpcpp

# Compiler Flags
# -fsycl: Necessary for dpcpp to link the correct SYCL/oneAPI runtimes
CXXFLAGS     := -std=c++17 -Ofast -march=native -mavx2 -mfma -fopenmp -fsycl
CXXFLAGS     += -MMD -MP
# $(DNNLROOT) is set automatically when you run 'module load Intel_oneAPI_Toolkit_2021.2'
CXXFLAGS     += -Iinclude -I$(DNNLROOT)/include

# Linker Flags
# Points to the optimized libraries provided by the module
LDFLAGS      := -L$(DNNLROOT)/lib
LDFLAGS      += -Wl,-rpath,$(DNNLROOT)/lib
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
build/%: src/main_%.cpp
	@mkdir -p build
	@echo "Building $@ from $<..."
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf build

# --- Dependency Inclusion ---
-include $(BINS:=.d)
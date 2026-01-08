# --- Configuration ---
CXX          := dpcpp

# The path to your GCC 9.1 module
GCC91_LIB    := /apps/Modules/apps/gcc91/lib64

# Compiler Flags
# We remove --gcc-toolchain to stop the "iostream not found" error
# We keep -fsycl to ensure the oneAPI runtime is linked correctly
CXXFLAGS     := -std=c++17 -Ofast -march=native -fopenmp -fsycl
CXXFLAGS     += -D_GLIBCXX_USE_CXX11_ABI=1
CXXFLAGS     += -MMD -MP
CXXFLAGS     += -Iinclude -I$(DNNLROOT)/include

# Linker Flags
# 1. Point to oneDNN libs
# 2. Point to GCC 9.1 libs (to fix the GLIBCXX_3.4.21 errors)
LDFLAGS      := -L$(DNNLROOT)/lib -L$(GCC91_LIB)
LDFLAGS      += -Wl,-rpath,$(DNNLROOT)/lib
LDFLAGS      += -Wl,-rpath,$(GCC91_LIB)
LDLIBS       := -ldnnl

# --- Auto-Discovery ---
SRCS := $(wildcard src/main_*.cpp)
BINS := $(patsubst src/main_%.cpp,build/%,$(SRCS))

# --- Targets ---
.PHONY: all clean

all: $(BINS)

build/%: src/main_%.cpp
	@mkdir -p build
	@echo "Building $@..."
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf build

-include $(BINS:=.d)
# --- Configuration ---
CXX          := dpcpp

# Unitn Hpc cluster path for gcc91 module
GCC_TOOLCHAIN := /apps/Modules/apps/gcc91

# Compiler Flags
# --gcc-toolchain: Forces dpcpp to use GCC 9.1 headers and ABI
# -D_GLIBCXX_USE_CXX11_ABI=1: Ensures string/list types match oneDNN's requirements
CXXFLAGS     := -std=c++17 -Ofast -march=native -fopenmp -fsycl
CXXFLAGS     += --gcc-toolchain=$(GCC_TOOLCHAIN)
CXXFLAGS     += -D_GLIBCXX_USE_CXX11_ABI=1
CXXFLAGS     += -MMD -MP
CXXFLAGS     += -Iinclude -I$(DNNLROOT)/include

# Linker Flags
# We include rpath for both oneDNN and the GCC 9.1 libraries so it runs correctly
LDFLAGS      := -L$(DNNLROOT)/lib -L$(GCC_TOOLCHAIN)/lib64
LDFLAGS      += --gcc-toolchain=$(GCC_TOOLCHAIN)
LDFLAGS      += -Wl,-rpath,$(DNNLROOT)/lib
LDFLAGS      += -Wl,-rpath,$(GCC_TOOLCHAIN)/lib64
LDLIBS       := -ldnnl

# --- Auto-Discovery ---
SRCS := $(wildcard src/main_*.cpp)
BINS := $(patsubst src/main_%.cpp,build/%,$(SRCS))

# --- Targets ---
.PHONY: all clean

all: $(BINS)

build/%: src/main_%.cpp
	@mkdir -p build
	@echo "Building $@ using GCC Toolchain at $(GCC_TOOLCHAIN)..."
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf build

-include $(BINS:=.d)
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <memory>
#include <cassert>
#include <omp.h>
#include <immintrin.h>
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

constexpr size_t ALIGNMENT = 64; // For avx2 and cache line alignment
constexpr int BENCH_REPEATS = 100;
constexpr int BENCH_WARMUP_REPEATS = 10;

// Round up a to the nearest multiple of b (a itself included)
constexpr inline int round_up(int a, int b) {
    return ((a + b - 1) / b) * b;
}

// Return a / b, rounded up to the nearest int
constexpr inline int ceil_division(int a, int b) {
    return (a + b - 1) / b;
}

using bfloat16_t = uint16_t;

inline bfloat16_t fp32_to_bf16(float f) {
    uint32_t input_bits = *reinterpret_cast<uint32_t*>(&f);
    uint32_t least_significant_bit = (input_bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + least_significant_bit;
    input_bits += rounding_bias;
    return static_cast<bfloat16_t>(input_bits >> 16);
}

inline float bf16_to_fp32(bfloat16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    return *reinterpret_cast<float*>(&bits);
}

enum class DType { FP32, BF16, INT8, INT32 };

struct Matrix {
    int rows;
    int cols;
    size_t dtype_size;
    DType dtype;
    char* raw_data; 

    Matrix(int r, int c, size_t d_sz, DType d_type) 
        : rows(r), cols(c), dtype_size(d_sz), dtype(d_type) {
        size_t bytes = rows * cols * dtype_size;
        raw_data = static_cast<char*>(std::aligned_alloc(ALIGNMENT, bytes));
        assert(raw_data != NULL && "Allocation failed");
        std::memset(raw_data, 0, bytes);
    }

    virtual ~Matrix() {
        std::free(raw_data);
    }
};

struct MatrixFP32 : public Matrix {
    MatrixFP32(int r, int c) : Matrix(r, c, sizeof(float), DType::FP32) {}

    float& at(int i, int j) {
        return reinterpret_cast<float*>(raw_data)[i * cols + j];
    }
    const float& at(int i, int j) const {
        return reinterpret_cast<const float*>(raw_data)[i * cols + j];
    }
};

struct MatrixBF16 : public Matrix {
    MatrixBF16(int r, int c) : Matrix(r, c, sizeof(bfloat16_t), DType::BF16) {}
    
    MatrixBF16(const MatrixFP32& src) : Matrix(src.rows, src.cols, sizeof(bfloat16_t), DType::BF16) {
        bfloat16_t* dst_ptr = reinterpret_cast<bfloat16_t*>(raw_data);
        for(int i=0; i < rows * cols; ++i) {
            dst_ptr[i] = fp32_to_bf16(reinterpret_cast<const float*>(src.raw_data)[i]);
        }
    }
    
    bfloat16_t& at(int i, int j) {
        return reinterpret_cast<bfloat16_t*>(raw_data)[i * cols + j];
    }
};

struct MatrixINT8 : public Matrix {
    float scale;
    int32_t zero_point;

    MatrixINT8(int r, int c) : Matrix(r, c, sizeof(int8_t), DType::INT8), scale(1.0f), zero_point(0) {}

    MatrixINT8(const MatrixFP32& src) : Matrix(src.rows, src.cols, sizeof(int8_t), DType::INT8) {
        const float* src_ptr = reinterpret_cast<const float*>(src.raw_data);
        int num_elements = rows * cols;

        float min_val = src_ptr[0];
        float max_val = src_ptr[0];
        for(int i=1; i<num_elements; ++i) {
            if(src_ptr[i] < min_val) min_val = src_ptr[i];
            if(src_ptr[i] > max_val) max_val = src_ptr[i];
        }

        scale = (max_val - min_val) / 255.0f;
        if (scale == 0.0f) scale = 1.0f; 
        
        zero_point = static_cast<int32_t>(std::round(-min_val / scale)) - 128;

        int8_t* dst_ptr = reinterpret_cast<int8_t*>(raw_data);
        for(int i=0; i<num_elements; ++i) {
            float val = src_ptr[i];
            int32_t q = static_cast<int32_t>(std::round(val / scale)) + zero_point;
            q = std::max(-128, std::min(127, q));
            dst_ptr[i] = static_cast<int8_t>(q);
        }
    }
};

struct MatrixINT32 : public Matrix {
    MatrixINT32(int r, int c) : Matrix(r, c, sizeof(int32_t), DType::INT32) {}
    
    int32_t& at(int i, int j) {
        return reinterpret_cast<int32_t*>(raw_data)[i * cols + j];
    }
};

MatrixFP32 load_matrix_fp32(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) { std::cerr << "Error: " << filename << std::endl; exit(1); }
    int rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    MatrixFP32 mat(rows, cols);
    file.read(mat.raw_data, rows * cols * sizeof(float));
    return mat;
}

MatrixFP32 generate_random_fp32(int rows, int cols) {
    MatrixFP32 mat(rows, cols);
    std::mt19937 gen(42);
    std::normal_distribution<float> d(0.0f, 1.0f); 
    float* ptr = reinterpret_cast<float*>(mat.raw_data);
    for (int i = 0; i < rows * cols; ++i) ptr[i] = d(gen);
    return mat;
}

void gemm_bf16_naive(const Matrix& A_base, const Matrix& B_base, Matrix& C_base) {
    const auto& A = static_cast<const MatrixBF16&>(A_base);
    const auto& B = static_cast<const MatrixBF16&>(B_base);
    auto& C = static_cast<MatrixFP32&>(C_base);

    const bfloat16_t* A_ptr = reinterpret_cast<const bfloat16_t*>(A.raw_data);
    const bfloat16_t* B_ptr = reinterpret_cast<const bfloat16_t*>(B.raw_data);
    float* C_ptr = reinterpret_cast<float*>(C.raw_data);

    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k) {
                sum += bf16_to_fp32(A_ptr[i * A.cols + k]) * bf16_to_fp32(B_ptr[k * B.cols + j]);
            }
            C_ptr[i * C.cols + j] = sum;
        }
    }
}

void gemm_int8_naive(const Matrix& A_base, const Matrix& B_base, Matrix& C_base) {
    const auto& A = static_cast<const MatrixINT8&>(A_base);
    const auto& B = static_cast<const MatrixINT8&>(B_base);
    auto& C = static_cast<MatrixINT32&>(C_base);

    const int8_t* A_ptr = reinterpret_cast<const int8_t*>(A.raw_data);
    const int8_t* B_ptr = reinterpret_cast<const int8_t*>(B.raw_data);
    int32_t* C_ptr = reinterpret_cast<int32_t*>(C.raw_data);

    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < A.cols; ++k) {
                sum += static_cast<int32_t>(A_ptr[i * A.cols + k]) * static_cast<int32_t>(B_ptr[k * B.cols + j]);
            }
            C_ptr[i * C.cols + j] = sum;
        }
    }
}

/*  Bf16 tiling sizes.

    For padding correctness we assume that:
    - M_CACHE_TILE_BF16 is a multiple of M_REGISTER_TILE_BF16
    - N_CACHE_TILE_BF16 is a multiple of N_REGISTER_TILE_BF16
    - K_CACHE_TILE_BF16 is a multiple of K_UNROLL_BF16
*/
constexpr int M_CACHE_TILE_BF16 = 256;
constexpr int N_CACHE_TILE_BF16 = 1024;
constexpr int K_CACHE_TILE_BF16 = 256;
constexpr int M_REGISTER_TILE_BF16 = 8; // avx2 has 16 YMM registers, we use 8 as accumulators, and the rest as tmp ones
constexpr int N_REGISTER_TILE_BF16 = 8; // 8 floats in 256bit avx2 registers
constexpr int K_UNROLL_BF16 = 1; // We will pad the k dimensions of cache level tiles to be divisible by this (we keep this low to avoid overhead for small K GEMM)
constexpr int PARALLEL_M_THRESHOLD = 64; // We parallelize over M only if M is big enough

/*  Micro kernel for bf16 GEMM.
    
    We will write to a [MR x n_remain] tile of C.

    We assume the following: 
    - MR == NR == 8.
    - B_packed is expected to be zero padded to a size divisible by NR = 8, to allow seamless avx2 loading from it.
    - The n_remain dimension can be any int from 1 to 8 inclusive, whereas MR=8 is fixed
*/
inline void microkernel_8x8_avx2_bf16(
    const bfloat16_t* __restrict__ A, 
    const bfloat16_t* __restrict__ B, 
    float* __restrict__ C, 
    int C_stride, int n_remain, int k_pad,
    bool first_k_block
) {
    constexpr int MR = M_REGISTER_TILE_BF16;
    constexpr int NR = N_REGISTER_TILE_BF16;

    static_assert(MR == 8, "microkernel_8x8_avx2_bf16 assumes MR == 8");
    static_assert(NR == 8, "microkernel_8x8_avx2_bf16 assumes NR == 8");

    // MR=8 output accumulators, each one containing NR=8 fp32 scalars. We will use only the first m_remain ones.
    __m256 C_acc[MR];

    if (first_k_block) {
        // If we are at the first accumulation iteration, init to zero the local accumulator 
        #pragma unroll(MR)
        for (int i = 0; i < MR; ++i) {
            C_acc[i] = _mm256_setzero_ps();
        }
    } else {
        // Otherwise we load from memory the partial sums
        if (n_remain == NR) {
            // When possible we load to the avx2 registers directly
            #pragma unroll(MR)
            for (int i = 0; i < MR; ++i) {
                C_acc[i] = _mm256_loadu_ps(&C[i * C_stride]);
            }
        } else {
            // Else we repeatedly load n_remain scalars into tmp, and then load with avx2 form tmp
            float tmp[NR];
            memset(tmp, 0, NR * sizeof(float));

            #pragma unroll(MR)
            for (int i = 0; i < MR; ++i) {
                for (int j = 0; j < n_remain; ++j) {
                    tmp[j] = C[i * C_stride + j];
                }
                C_acc[i] = _mm256_loadu_ps(tmp);
            }
        }
    }

    // Hot loop
    #pragma unroll(K_UNROLL_BF16)
    for (int k = 0; k < k_pad; ++k) {
        // Load 8 x 16bit scalars (bf16)
        __m128i v_b_bf16 = _mm_load_si128(reinterpret_cast<const __m128i*>(&B[k * NR]));

        // Convert from bf16 to fp32:
        // 1. We upcast each 16 bit lane to 32 bits, by extending on the left with zeros
        __m256i v_b_extended = _mm256_cvtepu16_epi32(v_b_bf16);
        // 2. We shift left each 32bit lane by 16, so the original 16bits are on the left (and we have zeros on the right)
        __m256 v_b = _mm256_castsi256_ps(_mm256_slli_epi32(v_b_extended, 16));

        #pragma unroll(MR)
        for (int i = 0; i < MR; ++i) {
            // Convert from bf16 to fp32:
            // 1. Read from memory a 16bit scalar 0xABCD and broadcast it to all the 16 lanes of width 16bits
            __m256i v_a_broadcast = _mm256_set1_epi16(A[k * MR + i]);
            // 2. Shift left by 16bits all the 32bit lanes (each lane goes from 0xABCDABCD to 0xABCD0000), and reinterpret as float
            __m256 v_a = _mm256_castsi256_ps(_mm256_slli_epi32(v_a_broadcast, 16));

            // Fused multiply add: C_ij = A_j * B_j + C_ij
            C_acc[i] = _mm256_fmadd_ps(v_a, v_b, C_acc[i]);
        }
    }

    // Write back to C
    if (n_remain == NR) {
        // When possible write result to the [m_remain x 8] tile of C, with avx2
        #pragma unroll(MR)
        for (int i = 0; i < MR; ++i) {
            _mm256_storeu_ps(&C[i * C_stride], C_acc[i]);
        }     
    } else {
        // Else we write with avx2 to tmp, and from tmp store only n_remain values
        float tmp[NR];
        for (int i = 0; i < MR; ++i) {
            _mm256_storeu_ps(tmp, C_acc[i]);
            for (int j = 0; j < n_remain; ++j) {
                C[i * C_stride + j] = tmp[j];
            }
        }
    }
}

/*  Cleanup micro kernel for bf16 GEMM.
    
    We will write to a [m_remain x n_remain] tile of C.
    B_packed is expected to be zero padded to a size divisible by NR = 8, to allow seamless avx2 loading from it.
    The m_remain and n_remain dimensions can be any int from 1 to 8 inclusive.

    We assume MR == NR == 8
*/
inline void microkernel_cleanup_avx2_bf16(
    const bfloat16_t* __restrict__ A, 
    const bfloat16_t* __restrict__ B, 
    float* __restrict__ C, 
    int C_stride, int m_remain, int n_remain, int k_pad,
    bool first_k_block
) {
    constexpr int MR = M_REGISTER_TILE_BF16;
    constexpr int NR = N_REGISTER_TILE_BF16;

    static_assert(MR == 8, "microkernel_cleanup_avx2_bf16 assumes MR == 8");
    static_assert(NR == 8, "microkernel_cleanup_avx2_bf16 assumes NR == 8");

    // MR=8 output accumulators, each one containing NR=8 fp32 scalars. We will use only the first m_remain ones.
    __m256 C_acc[MR];

    // In all the i loops we suggest the compiler to unroll by MR, since m_remain == MR == 8 is the 
    // most common scenario in a GEMM with big enough M. Likely the compiler will generate unrolled
    // loops with 8 copies of the loop body, and a cleanup loop afterwards.

    if (first_k_block) {
        // If we are at the first accumulation iteration, init to zero the local accumulator 
        for (int i = 0; i < m_remain; ++i) {
            C_acc[i] = _mm256_setzero_ps();
        }
    } else {
        // Otherwise we load from memory the partial sums
        if (n_remain == NR) {
            // When possible we load to the avx2 registers directly
            for (int i = 0; i < m_remain; ++i) {
                C_acc[i] = _mm256_loadu_ps(&C[i * C_stride]);
            }
        } else {
            // Else we repeatedly load n_remain scalars into tmp, and then load with avx2 form tmp
            float tmp[NR];
            memset(tmp, 0, NR * sizeof(float));

            for (int i = 0; i < m_remain; ++i) {
                for (int j = 0; j < n_remain; ++j) {
                    tmp[j] = C[i * C_stride + j];
                }
                C_acc[i] = _mm256_loadu_ps(tmp);
            }
        }
    }

    // Hot loop
    #pragma unroll(K_UNROLL_BF16)
    for (int k = 0; k < k_pad; ++k) {
        // Load 8 x 16bit scalars (bf16)
        __m128i v_b_bf16 = _mm_load_si128(reinterpret_cast<const __m128i*>(&B[k * NR]));

        // Convert from bf16 to fp32:
        // 1. We upcast each 16 bit lane to 32 bits, by extending on the left with zeros
        __m256i v_b_extended = _mm256_cvtepu16_epi32(v_b_bf16);
        // 2. We shift left each 32bit lane by 16, so the original 16bits are on the left (and we have zeros on the right)
        __m256 v_b = _mm256_castsi256_ps(_mm256_slli_epi32(v_b_extended, 16));

        for (int i = 0; i < m_remain; ++i) {
            // Convert from bf16 to fp32:
            // 1. Read from memory a 16bit scalar 0xABCD and broadcast it to all the 16 lanes of width 16bits
            __m256i v_a_broadcast = _mm256_set1_epi16(A[k * m_remain + i]);
            // 2. Shift left by 16bits all the 32bit lanes (each lane goes from 0xABCDABCD to 0xABCD0000), and reinterpret as float
            __m256 v_a = _mm256_castsi256_ps(_mm256_slli_epi32(v_a_broadcast, 16));

            // Fused multiply add: C_ij = A_j * B_j + C_ij
            C_acc[i] = _mm256_fmadd_ps(v_a, v_b, C_acc[i]);
        }
    }

    // Write back to C
    if (n_remain == NR) {
        // When possible write result to the [m_remain x 8] tile of C, with avx2
        for (int i = 0; i < m_remain; ++i) {
            _mm256_storeu_ps(&C[i * C_stride], C_acc[i]);
        }     
    } else {
        // Else we write with avx2 to tmp, and from tmp store only n_remain values
        float tmp[NR];
        for (int i = 0; i < m_remain; ++i) {
            _mm256_storeu_ps(tmp, C_acc[i]);
            for (int j = 0; j < n_remain; ++j) {
                C[i * C_stride + j] = tmp[j];
            }
        }
    }
}

/*  Read a 8x8 block of bfloat16_t (uint16_t) from src matrix, transpose it with avx2 and write it to dst matrix.

    The transpose is performed as follows:
    - At first we have 8 vector registers, each one containing a row of the matrix (8 scalars)
    - We want to end up with 8 vector registers containing the columns of the matrix

    Conceptually, to achieve this we "split" and "merge" recursively:
    - Start from 8 [1 x 8] blocks (these are the initial rows, each one stored in a register)
    - Stage 1: Split in 2 the blocks horizontally and merge pairs vertically into 2 new [2 x 4] "column major" blocks (each one stored in a register)
    - Stage 2: Split in 2 the blocks horizontally and merge pairs vertically into 2 new [4 x 2] "column major" blocks (each one stored in a register)
    - Stage 3: Split in 2 the blocks horizontally and merge pairs vertically into 2 new [8 x 1] "column major" blocks 
        (these are the final columns, each one stored in a register)

    Each "horizontal split" + "vertical merge" is done by unpacking and interleaving the columns of 
    the blocks stored in the registers. In practice, this means that:
    - Stage 1: we unpack and interleave groups of 16 bits ("columns" of 1 scalar)
    - Stage 2: groups of 32 bits (columns of 2 scalars) 
    - Stage 3: groups of 64 bits (columns of 4 scalars) 

    In this way we keep a "column major" layout at each step. This leads at the end to the columns 
    being stored contiguously in 8 registers, which we can finally write directly to dst.
*/
inline void transpose_8x8_avx2_bf16(
    const bfloat16_t* __restrict__ src, 
    bfloat16_t* __restrict__ dst, 
    int src_stride, 
    int dst_stride
) {
    // Here we declare 12 avx2 registers (out of 16)
    // We will explicitly reuse these registers (to avoid spilling)
    __m128i a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3; 

    // Load 8x8 block from src to avx2 registers (8 avx2 registers, each one containing 8 x bf16)
    a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 0]));
    a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 1]));
    a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 2]));
    a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 3]));
    a4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 4]));
    a5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 5]));
    a6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 6]));
    a7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 7]));

    // Stage 1: unpack and interleave groups of 16bits (1 scalar)
    b0 = _mm_unpacklo_epi16(a0, a1);
    b1 = _mm_unpackhi_epi16(a0, a1);
    b2 = _mm_unpacklo_epi16(a2, a3);
    b3 = _mm_unpackhi_epi16(a2, a3);
    a0 = _mm_unpacklo_epi16(a4, a5); // Here we start reusing the free registers (from now on we keep "cycling" them, to avoid spilling)
    a1 = _mm_unpackhi_epi16(a4, a5);
    a2 = _mm_unpacklo_epi16(a6, a7);
    a3 = _mm_unpackhi_epi16(a6, a7);

    // Stage 2: unpack and interleave groups of 32bits (2 scalars)
    a4 = _mm_unpacklo_epi32(b0, b2);
    a5 = _mm_unpackhi_epi32(b0, b2);
    a6 = _mm_unpacklo_epi32(b1, b3);
    a7 = _mm_unpackhi_epi32(b1, b3);
    b0 = _mm_unpacklo_epi32(a0, a2);
    b1 = _mm_unpackhi_epi32(a0, a2);
    b2 = _mm_unpacklo_epi32(a1, a3);
    b3 = _mm_unpackhi_epi32(a1, a3);

    // Stage 3: unpack and interleave groups of 64bits (4 scalars)
    a0 = _mm_unpacklo_epi64(a4, b0);
    a1 = _mm_unpackhi_epi64(a4, b0); 
    a2 = _mm_unpacklo_epi64(a5, b1);
    a3 = _mm_unpackhi_epi64(a5, b1);
    a4 = _mm_unpacklo_epi64(a6, b2);
    a5 = _mm_unpackhi_epi64(a6, b2);
    b0 = _mm_unpacklo_epi64(a7, b3);
    b1 = _mm_unpackhi_epi64(a7, b3);

    // We store the result to the 8x8 block of dst
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 0]), a0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 1]), a1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 2]), a2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 3]), a3);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 4]), a4);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 5]), a5);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 6]), b0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 7]), b1);
}

/*  Pack a cache level tile of A into the A_packed buffer using a layout optimized for the microkernel.

    We employ the following nested layout so the data can be read contiguously by the microkernel: 
    - A_packed contains ceil(m_end / MR) micropanels (i_register loop outside the microkernel)
    - each micropanel contains k_pad columns (k loop in the microkernel)
    - each column contains m_remain values (i loop in the microkernel); m_remain is always MR for all the columns, 
        except for the columns of the last micropanel, where m_remain can be any int from 1 to MR

    The data of A_packed is accessed as follows: 
        A_packed[i_register * micropanel_stride + k * m_remain + i]

    As said before, m_remain == MR in general, except possibly for the last micropanel.

    We assume:
    - MR == M_REGISTER_TILE_BF16 == 8
    - sizeof(bfloat16_t) == 2 bytes
    - A_packed is allocated with aligned_alloc(ALIGNMENT, ...)
    - A_packed has at least k_pad * m_end * sizeof(bfloat16_t) bytes of allocated memory
    - micropanel_stride is a multiple of MR and of ALIGNMENT / sizeof(bfloat16_t)
*/
inline void packA_avx2_bf16(
    const bfloat16_t *__restrict__ A_tile, /* Pointer to the start of the B cache level tile */
    bfloat16_t *__restrict__ A_packed,     /* Destination buffer */
    int A_stride, /* Offset between the start of two rows of A_tile */
    int micropanel_stride, /* Offset between the start of two consecutive micropanels in A_packed */
    int m_end, int k_end, int k_pad
) {
    constexpr int MR = M_REGISTER_TILE_BF16;
    static_assert(MR == 8, "packA_avx2_bf16 assumes MR == 8");

    int i_register = 0;

    // Main packing loop (with avx2, m_remain == MR == 8)
    for (; (i_register + 1) * MR <= m_end; ++i_register) {
        int k = 0;

        // First we iterate through MRxMR (8x8) blocks of A_tile, transpose them with avx2 and write them to A_packed
        for (; k + MR <= k_end; k += MR) {
            const auto* src = &A_tile[(i_register * MR) * A_stride + k];
            auto* dst = &A_packed[i_register * micropanel_stride + k * MR];
            transpose_8x8_avx2_bf16(src, dst, A_stride, MR); // Here the dst matrix is a MRxMR block inside a micropanel
        }

        // Cleanup k (no avx)
        for (; k < k_end; ++k) {
            const auto* src = &A_tile[(i_register * MR) * A_stride + k];
            auto* dst = &A_packed[i_register * micropanel_stride + k * MR];
            for (int i = 0; i < MR; ++i) {
                dst[i] = src[i * A_stride];
            }
        }

        // Zero pad along k
        for (; k < k_pad; ++k) {
            auto* dst = &A_packed[i_register * micropanel_stride + k * MR];
            _mm_store_si128(reinterpret_cast<__m128i*>(dst), _mm_setzero_si128());
        }
    }

    // Cleanup i (no avx, m_remain < 8)
    int m_remain = m_end - i_register * MR;
    int k = 0;
    for (; k < k_end; ++k) {
        const auto* src = &A_tile[(i_register * MR) * A_stride + k];
        auto* dst = &A_packed[i_register * micropanel_stride + k * m_remain];
        for (int i = 0; i < m_remain; ++i) {
            dst[i] = src[i * A_stride];
        }
    }
    for (; k < k_pad; ++k) {
        auto* dst = &A_packed[i_register * micropanel_stride + k * m_remain];
        for (int i = 0; i < m_remain; ++i) {
            dst[i] = 0;
        }
    }
}

/*  Pack a cache level tile of B into the B_packed buffer using a layout optimized for the microkernel.

    We employ the following nested layout so the data can be read contiguously by the microkernel: 
    - B_packed contains ceil(n_end / NR) micropanels (j_register loop outside the microkernel)
    - each micropanel contains k_pad rows (k loop in the microkernel)
    - each row contains NR values (avx2 vectorization in the microkernel)

    The memory layout of B_packed is accessed as follows:
        B_packed[j_register * micropanel_stride + k * NR + j]

    We will employ zero padding to ensure that even the rows of the last micropanel contain exactly NR values.

    We assume:
    - NR == N_REGISTER_TILE_BF16 == 8
    - sizeof(bfloat16_t) == 2 bytes
    - B_packed is allocated with aligned_alloc(ALIGNMENT, ...)
    - B_packed has at least k_pad * round_up(n_end, NR) * sizeof(bfloat16_t) bytes of allocated memory
    - micropanel_stride is a multiple of NR and of ALIGNMENT / sizeof(bfloat16_t)
*/
inline void packB_avx2_bf16(
    const bfloat16_t *__restrict__ B_tile, /* Pointer to the start of the B cache level tile */
    bfloat16_t *__restrict__ B_packed,     /* Destination buffer */
    int B_stride, /* Offset between the start of two consecutive rows in B_tile */
    int micropanel_stride, /* Offset between the start of two consecutive micropanels in B_packed */
    int n_end, int k_end, int k_pad
) {
    constexpr int NR = N_REGISTER_TILE_BF16;
    static_assert(NR == 8, "packB_avx2_bf16 assumes NR == 8");

    // Main packing loop (SIMD)
    int j_register = 0; 
    for (; (j_register + 1) * NR <= n_end; ++j_register) {
        int k = 0;
        for (; k < k_end; ++k) {
            const auto* src = &B_tile[k * B_stride + (j_register * NR)];
            auto* dst = &B_packed[j_register * micropanel_stride + k * NR];
            // Load from src 8 x 16bit scalars, and then store them to dst
            __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
            _mm_store_si128(reinterpret_cast<__m128i*>(dst), tmp);
        }

        // Pad to zero for all remaining k
        for (; k < k_pad; ++k) {
            auto* dst = &B_packed[j_register * micropanel_stride + k * NR];
            _mm_store_si128(reinterpret_cast<__m128i*>(dst), _mm_setzero_si128());
        }
    }

    // Now we handle the remaining j up to n_end (this can be viewed as a last step in the j_register loop).
    // In other words, we are handling the last micropanel if there are some remaining j
    int n_remain = n_end - j_register * NR;
    if (n_remain == 0) return; 

    // We set directly to zero the whole last micropanel to handle zero-padding.
    // We can set to zero with simd NR > n_remain values since we assumed to own enough memory.
    for (int k = 0; k < k_pad; ++k) {
        auto* dst = &B_packed[j_register * micropanel_stride + k * NR];
        _mm_store_si128(reinterpret_cast<__m128i*>(dst), _mm_setzero_si128());
    }

    // Read the remaining scalars from src and write them to dst
    for (int k = 0; k < k_end; ++k) {
        for (int j = 0; j < n_remain; ++j) {
            auto src_val = B_tile[k * B_stride + (j_register * NR + j)];
            B_packed[j_register * micropanel_stride + k * NR + j] = src_val;
        }
    }
}

inline void gemm_bf16_avx2_tiled_parallel_N(
    const bfloat16_t* __restrict__ A, 
    const bfloat16_t* __restrict__ B, 
    float* __restrict__ C, 
    const int M, const int N, const int K
) {
    const int num_threads = omp_get_max_threads();
    const int MR = M_REGISTER_TILE_BF16;
    const int NR = N_REGISTER_TILE_BF16;
    
    static_assert(NR == 8, "gemm_bf16_avx2_tiled_parallel_N assumes NR == 8");
    static_assert(MR == 8, "gemm_bf16_avx2_tiled_parallel_N assumes MR == 8");
    assert(N > 0 && "N should be > 0");
    assert(M > 0 && "M should be > 0");
    assert(K > 0 && "K should be > 0");

    const int KC = round_up(std::min(K_CACHE_TILE_BF16, K), K_UNROLL_BF16); // We enforce KC to be a nonzero multiple of K_UNROLL_BF16
    const int MC = std::min(M_CACHE_TILE_BF16, M); // Note that we handle also MC < MR
    const int NC = round_up(std::clamp(N / num_threads, NR, N_CACHE_TILE_BF16), NR); // We enforce NC to be a nonzero multiple of NR

    #pragma omp parallel
    {
        // Allocate A_packed, private per thread (ideally stored in L2/L1 cache)
        const int max_micropanel_stride_A = round_up(KC * MR, ALIGNMENT / sizeof(bfloat16_t));
        const int n_micropanels_A = ceil_division(MC, MR);
        size_t elements_A_packed = size_t(n_micropanels_A) * max_micropanel_stride_A;
        size_t bytes_A_packed = round_up(n_micropanels_A * max_micropanel_stride_A * sizeof(bfloat16_t), ALIGNMENT);
        auto* A_packed = reinterpret_cast<bfloat16_t*>(std::aligned_alloc(ALIGNMENT, bytes_A_packed));

        // Allocate B_packed, private per thread (stored in L1/L2/L3 cache depending on N and on num_threads)
        const int max_micropanel_stride_B = round_up(KC * NR, ALIGNMENT / sizeof(bfloat16_t));
        const int n_micropanels_B = ceil_division(NC, NR);
        size_t elements_B_packed = size_t(n_micropanels_B) * max_micropanel_stride_B;
        size_t bytes_B_packed = round_up(elements_B_packed * sizeof(bfloat16_t), ALIGNMENT);
        auto* B_packed = reinterpret_cast<bfloat16_t*>(std::aligned_alloc(ALIGNMENT, bytes_B_packed));

        // (j_cache < j_cache_end) here is equivalent to (j_cache * NC < N)
        int j_cache_end = ceil_division(N, NC);

        // Cache level tiling 
        // j_cache * NC, k_cache * KC and i_cache * MC are "the starting indicies of cache level blocks"
        #pragma omp for schedule(static)
        for (int j_cache = 0; j_cache < j_cache_end; ++j_cache) {
            int n_end = std::min(NC, N - j_cache * NC);

            for (int k_cache = 0; k_cache * KC < K; ++k_cache) {
                // When packing A and B, we will pad along the k dim with zeros up to k_pad.
                // This ensures k_pad is divisible by the unrolling depth.
                const int k_end = std::min(KC, K - k_cache * KC);
                const int k_pad = round_up(k_end, K_UNROLL_BF16);

                // Stride between the start of two consecutive micropanels inside A_packed.
                // We round it up to ensure alignment.
                // See packA_avx2_bf16 for the A_packed layout explanation.
                const int micropanel_stride_A = round_up(k_pad * MR, ALIGNMENT / sizeof(bfloat16_t));

                // Stride between the start of two consecutive micropanels inside B_packed.
                // We round it up to ensure alignment.
                // See packB_avx2_bf16() for the B_packed layout explanation.
                const int micropanel_stride_B = round_up(k_pad * NR, ALIGNMENT / sizeof(bfloat16_t));

                // Each thread packs a cache level tile of B into a B_packed private buffer
                packB_avx2_bf16(
                    &B[(k_cache * KC) * N + (j_cache * NC)], B_packed, 
                    N, micropanel_stride_B, n_end, k_end, k_pad
                );

                // (i_cache < i_cache_end) here is equivalent to (i_cache * MC < M)
                int i_cache_end = ceil_division(M, MC);
                for (int i_cache = 0; i_cache < i_cache_end; ++i_cache) {
                    int m_end = std::min(MC, M - i_cache * MC);

                    // Each thread packs a cache level tile of A into a A_packed private buffer
                    packA_avx2_bf16(
                        &A[(i_cache * MC) * K + (k_cache * KC)], A_packed, 
                        K, micropanel_stride_A, m_end, k_end, k_pad
                    );
            
                    // Register level tiling
                    // i_register * MR and j_register * NR are "the starting offsets of register level blocks inside the cache level blocks"
                    int i_register = 0;
                    for (; (i_register + 1) * MR <= m_end; ++i_register) {
                        for (int j_register = 0; j_register * NR < n_end; ++j_register) {
                            int n_remain = std::min(NR, n_end - j_register * NR);
                            microkernel_8x8_avx2_bf16(
                                &A_packed[i_register * micropanel_stride_A],
                                &B_packed[j_register * micropanel_stride_B],
                                &C[(i_cache * MC + i_register * MR) * N + (j_cache * NC + j_register * NR)],
                                N, n_remain, k_pad,
                                (k_cache == 0)
                            );
                        }
                    }
                    
                    // Cleanup in case m_end was not divisible by MR
                    int m_remain = std::min(MR, m_end - i_register * MR);
                    if (m_remain != 0) {
                        for (int j_register = 0; j_register * NR < n_end; ++j_register) {
                            int n_remain = std::min(NR, n_end - j_register * NR);
                            microkernel_cleanup_avx2_bf16(
                                &A_packed[i_register * micropanel_stride_A],
                                &B_packed[j_register * micropanel_stride_B],
                                &C[(i_cache * MC + i_register * MR) * N + (j_cache * NC + j_register * NR)],
                                N, m_remain, n_remain, k_pad,
                                (k_cache == 0)
                            );
                        }
                    }
                }
            }
        }

        std::free(A_packed);
        std::free(B_packed);
    }
}

inline void gemm_bf16_avx2_tiled_parallel_M(
    const bfloat16_t* __restrict__ A, 
    const bfloat16_t* __restrict__ B, 
    float* __restrict__ C, 
    const int M, const int N, const int K
) {
    const int num_threads = omp_get_max_threads();
    const int MR = M_REGISTER_TILE_BF16;
    const int NR = N_REGISTER_TILE_BF16;
    
    static_assert(NR == 8, "gemm_bf16_avx2_tiled_parallel_M assumes NR == 8");
    static_assert(MR == 8, "gemm_bf16_avx2_tiled_parallel_M assumes MR == 8");
    assert(N > 0 && "N should be > 0");
    assert(M > 0 && "M should be > 0");
    assert(K > 0 && "K should be > 0");

    const int KC = round_up(std::min(K_CACHE_TILE_BF16, K), K_UNROLL_BF16); // We enforce KC to be a nonzero multiple of K_UNROLL_BF16
    const int MC = std::clamp(M / num_threads, 1, M_CACHE_TILE_BF16); // Note that we handle also MC < MR
    const int NC = round_up(std::min(N, N_CACHE_TILE_BF16), NR); // We enforce NC to be a nonzero multiple of NR

    // Allocate B_packed, shared across threads (ideally stored in L3 cache)
    const int max_micropanel_stride_B = round_up(KC * NR, ALIGNMENT / sizeof(bfloat16_t));
    const int n_micropanels_B = ceil_division(NC, NR);
    size_t elements_B_packed = size_t(n_micropanels_B) * max_micropanel_stride_B;
    size_t bytes_B_packed = round_up(elements_B_packed * sizeof(bfloat16_t), ALIGNMENT);
    auto* B_packed = reinterpret_cast<bfloat16_t*>(std::aligned_alloc(ALIGNMENT, bytes_B_packed));

    #pragma omp parallel
    {
        // Allocate A_packed, private per thread (ideally stored in L2/L1 cache)
        const int max_micropanel_stride_A = round_up(KC * MR, ALIGNMENT / sizeof(bfloat16_t));
        const int n_micropanels_A = ceil_division(MC, MR);
        size_t elements_A_packed = size_t(n_micropanels_A) * max_micropanel_stride_A;
        size_t bytes_A_packed = round_up(n_micropanels_A * max_micropanel_stride_A * sizeof(bfloat16_t), ALIGNMENT);
        auto* A_packed = reinterpret_cast<bfloat16_t*>(std::aligned_alloc(ALIGNMENT, bytes_A_packed));

        // Cache level tiling 
        // j_cache * NC, k_cache * KC and i_cache * MC are "the starting indicies of cache level blocks"
        for (int j_cache = 0; j_cache * NC < N; ++j_cache) {
            int n_end = std::min(NC, N - j_cache * NC);

            for (int k_cache = 0; k_cache * KC < K; ++k_cache) {
                // When packing A and B, we will pad along the k dim with zeros up to k_pad.
                // This ensures k_pad is divisible by the unrolling depth.
                const int k_end = std::min(KC, K - k_cache * KC);
                const int k_pad = round_up(k_end, K_UNROLL_BF16);

                // Stride between the start of two consecutive micropanels inside A_packed.
                // We round it up to ensure alignment.
                // See packA_avx2_bf16 for the A_packed layout explanation.
                const int micropanel_stride_A = round_up(k_pad * MR, ALIGNMENT / sizeof(bfloat16_t));

                // Stride between the start of two consecutive micropanels inside B_packed.
                // We round it up to ensure alignment.
                // See packB_avx2_bf16() for the B_packed layout explanation.
                const int micropanel_stride_B = round_up(k_pad * NR, ALIGNMENT / sizeof(bfloat16_t));

                // Pack a cache level tile of B with a single thread into a B_packed buffer shared across threads
                #pragma omp single 
                packB_avx2_bf16(
                    &B[(k_cache * KC) * N + (j_cache * NC)], B_packed, 
                    N, micropanel_stride_B, n_end, k_end, k_pad
                );

                // (i_cache < i_cache_end) here is equivalent to (i_cache * MC < M)
                int i_cache_end = ceil_division(M, MC);

                #pragma omp for schedule(static)
                for (int i_cache = 0; i_cache < i_cache_end; ++i_cache) {
                    int m_end = std::min(MC, M - i_cache * MC);

                    // Each thread packs a cache level tile of A into a A_packed private buffer
                    packA_avx2_bf16(
                        &A[(i_cache * MC) * K + (k_cache * KC)], A_packed, 
                        K, micropanel_stride_A, m_end, k_end, k_pad
                    );
            
                    // Register level tiling
                    // i_register * MR and j_register * NR are "the starting offsets of register level blocks inside the cache level blocks"
                    int i_register = 0;
                    for (; (i_register + 1) * MR <= m_end; ++i_register) {
                        for (int j_register = 0; j_register * NR < n_end; ++j_register) {
                            int n_remain = std::min(NR, n_end - j_register * NR);
                            microkernel_8x8_avx2_bf16(
                                &A_packed[i_register * micropanel_stride_A],
                                &B_packed[j_register * micropanel_stride_B],
                                &C[(i_cache * MC + i_register * MR) * N + (j_cache * NC + j_register * NR)],
                                N, n_remain, k_pad,
                                (k_cache == 0)
                            );
                        }
                    }
                    
                    // Cleanup in case m_end was not divisible by MR
                    int m_remain = std::min(MR, m_end - i_register * MR);
                    if (m_remain != 0) {
                        for (int j_register = 0; j_register * NR < n_end; ++j_register) {
                            int n_remain = std::min(NR, n_end - j_register * NR);
                            microkernel_cleanup_avx2_bf16(
                                &A_packed[i_register * micropanel_stride_A],
                                &B_packed[j_register * micropanel_stride_B],
                                &C[(i_cache * MC + i_register * MR) * N + (j_cache * NC + j_register * NR)],
                                N, m_remain, n_remain, k_pad,
                                (k_cache == 0)
                            );
                        }
                    }
                }
            }
        }

        std::free(A_packed);
    }

    std::free(B_packed);
}

void gemm_bf16_tiled(const Matrix& A_base, const Matrix& B_base, Matrix& C_base) {
    const auto& A = static_cast<const MatrixBF16&>(A_base);
    const auto& B = static_cast<const MatrixBF16&>(B_base);
    auto& C = static_cast<MatrixFP32&>(C_base);

    const int M = A.rows;
    const int K = A.cols;
    const int N = B.cols;

    const auto* A_ptr = reinterpret_cast<const bfloat16_t*>(A.raw_data);
    const auto* B_ptr = reinterpret_cast<const bfloat16_t*>(B.raw_data);
    auto* C_ptr = reinterpret_cast<float*>(C.raw_data);

    // return gemm_bf16_avx2_tiled_parallel_M(A_ptr, B_ptr, C_ptr, M, N, K);
    return gemm_bf16_avx2_tiled_parallel_N(A_ptr, B_ptr, C_ptr, M, N, K);
}



// Int8 GEMM tiling sizes
constexpr int M_CACHE_INT8 = 768; 
constexpr int N_CACHE_INT8 = 128;
constexpr int K_CACHE_INT8 = 256;
constexpr int M_REGISTER_INT8 = 4;
constexpr int N_REGISTER_INT8 = 8;

inline void micro_kernel_int8_tiled(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    __m256i accumulator[M_REGISTER_INT8];
    for (int i = 0; i < M_REGISTER_INT8; ++i) {
        accumulator[i] = _mm256_load_epi32(C + i*N);
    }

    // We increase k with step 2 because we reduce horizontally 2 elements each time
    for (int k = 0; k < K_CACHE_INT8; k+=2) {

    }
}

inline void gemm_int8_tiled(const Matrix& A_base, const Matrix& B_base, Matrix& C_base) {
    // TODO: implement gemm_int8_tiled
    // std::cout << "warning: gemm_int8_tiled not implemented" << std::endl;
    return;

    const auto& A = static_cast<const MatrixINT8&>(A_base);
    const auto& B = static_cast<const MatrixINT8&>(B_base);
    // We accumulate in int32 to avoid overflows
    auto& C = static_cast<MatrixINT32&>(C_base); 

    // A is M x K
    // B is K x N
    // C is M x N
    
    int M = A.rows;
    int N = B.cols;
    int K = A.cols;

    const int8_t* A_ptr = reinterpret_cast<const int8_t*>(A.raw_data);
    const int8_t* B_ptr = reinterpret_cast<const int8_t*>(B.raw_data);
    int32_t* C_ptr = reinterpret_cast<int32_t*>(C.raw_data);

    // 3 loops for cache level tiling.
    // ii loop: Find vertical position of horizontal strip (A and C)
    // jj loop: Find horizontal position of vertical strip (B and C)
    // (Now we uniquely identified the output block of C)
    // kk loop: Finally loop over blocks inside the input strips (A and B)
    #pragma omp parallel for schedule(static) collapse(2)
    for (int ii = 0; ii < M; ii += M_CACHE_INT8) {
        for (int jj = 0; jj < N; jj += N_CACHE_INT8) {
            for (int kk = 0; kk < K; kk += K_CACHE_INT8) {

                int n_end = std::min(M_CACHE_INT8, M - ii);
                int m_end = std::min(N_CACHE_INT8, N - jj);
                int k_end = std::min(K_CACHE_INT8, K - kk);

                // We will pad to the highest multiple of the register level tiling sizes
                int m_pad = round_up(m_end, M_REGISTER_INT8);
                int n_pad = round_up(n_end, N_REGISTER_INT8);
                
                // We pack the cache level blocks, so that the microkernel can read data contiguously
                // During packing we also pad the data to have sizes that align with the register level tiling sizes
                alignas(ALIGNMENT) int8_t A_packed[M_CACHE_INT8 * K_CACHE_INT8];
                alignas(ALIGNMENT) int8_t B_packed[K_CACHE_INT8 * N_CACHE_INT8];

                // Pack A block
                for (int i = 0; i < m_pad; ++i) {
                    for (int k = 0; k < n_pad; ++k) {
                        if (i >= m_end || k >= k_end) {
                            A_packed[i * N_CACHE_INT8 + k] = A_ptr[i];
                        } else {
                            A_packed[i * N_CACHE_INT8 + k] = 0;
                        }
                    }
                }


                // Pack B block
                // Here we interleave data such that elements from k and k+1 are contiguous

                
                // 2 loops for register level blocks
                for (int i = 0; i < M_CACHE_INT8; i += M_REGISTER_INT8) {
                    for (int j = 0; j < jj + N_CACHE_INT8; j += N_REGISTER_INT8) {
                        micro_kernel_int8_tiled(
                            A_ptr + i*K + kk, 
                            B_ptr + kk*N + j, 
                            C_ptr + i*N + j,
                            M, N, K
                        );
                    }


                }
            }
        }
    }
}


void gemm_onednn(const Matrix& A, const Matrix& B, Matrix& C) {
    engine eng(engine::kind::cpu, 0);
    stream engine_stream(eng);

    memory::dims a_dims = {A.rows, A.cols};
    memory::dims b_dims = {B.rows, B.cols};
    memory::dims c_dims = {C.rows, C.cols};

    auto get_dnnl_type = [](DType dt) {
        switch(dt) {
            case DType::FP32: return memory::data_type::f32;
            case DType::BF16: return memory::data_type::bf16;
            case DType::INT8: return memory::data_type::s8; 
            case DType::INT32: return memory::data_type::s32;
            default: return memory::data_type::f32;
        }
    };

    auto a_md = memory::desc(a_dims, get_dnnl_type(A.dtype), memory::format_tag::ab);
    auto b_md = memory::desc(b_dims, get_dnnl_type(B.dtype), memory::format_tag::ab);
    auto c_md = memory::desc(c_dims, get_dnnl_type(C.dtype), memory::format_tag::ab);

    auto a_mem = memory(a_md, eng, A.raw_data);
    auto b_mem = memory(b_md, eng, B.raw_data);
    auto c_mem = memory(c_md, eng, C.raw_data);

    auto matmul_d = matmul::primitive_desc(eng, a_md, b_md, c_md);
    auto matmul_prim = matmul(matmul_d);

    matmul_prim.execute(engine_stream, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem}
    });
    engine_stream.wait();
}


double benchmark_gemm(
    std::function<void(const Matrix&, const Matrix&, Matrix&)> func, 
    const Matrix& A, const Matrix& B, Matrix& C
) {
    // Warm-up (discarded)
    for (int i = 0; i < BENCH_WARMUP_REPEATS; ++i) {
        func(A, B, C); // Warmup
    }
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < BENCH_REPEATS; ++i) {
        func(A, B, C);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count() / BENCH_REPEATS;
}

// double benchmark_gemm (
//     std::function<void(const Matrix&, const Matrix&, Matrix&)> func,
//     const Matrix& A, const Matrix& B, Matrix& C
// ) {
//     // Warm-up (discarded)
//     for (int i = 0; i < BENCH_WARMUP_REPEATS; ++i) {
//         func(A, B, C);
//     }

//     // Timed runs 
//     std::vector<double> times(BENCH_REPEATS);
//     for (int i = 0; i < BENCH_REPEATS; ++i) {
//         auto start = std::chrono::high_resolution_clock::now();
//         func(A, B, C);
//         auto end = std::chrono::high_resolution_clock::now();

//         std::chrono::duration<double> diff = end - start;
//         times[i] = diff.count();
//     }

//     // Compute median
//     std::sort(times.begin(), times.end());
//     int mid = BENCH_REPEATS / 2;

//     if (BENCH_REPEATS % 2 == 0) {
//         return 0.5 * (times[mid - 1] + times[mid]);
//     } else {
//         return times[mid];
//     }
// }


// Benchmarks our tiled BF16 gemm implementation (optionally compares to FP32 Baseline and/or naive implementation)
void run_bench_gemm_bf16(int M, const MatrixFP32& B_ref, bool run_naive = false, bool run_baseline = false, bool verbose = false) {
    MatrixFP32 A_fp32 = generate_random_fp32(M, B_ref.rows); // Original FP32 Input
    MatrixBF16 A_bf16(A_fp32); // Converted Input
    MatrixBF16 B_bf16(B_ref);  // Converted Weights
    
    MatrixFP32 C_tiled(M, B_ref.cols);

    double ops = 2.0 * M * B_ref.cols * B_ref.rows * 1e-9;

    // Run BF16 Tiled (Emulated)
    double t_tiled = benchmark_gemm(gemm_bf16_tiled, A_bf16, B_bf16, C_tiled);
    std::cout << "BF16 Tiled:             " << t_tiled << "s | " << ops/t_tiled << " GFLOPS" << std::endl;


    // Tiled vs BASELINE (OneDNN FP32) comparison
    // We use the FP32 matrices here to establish the "Hardware limit" for standard precision
    if (run_baseline) {
        MatrixFP32 C_onednn_fp32(M, B_ref.cols); // Output for Baseline
        double t_base = benchmark_gemm(gemm_onednn, A_fp32, B_ref, C_onednn_fp32);
        std::cout << "BASELINE (FP32 OneDNN): " << t_base << "s | " << ops/t_base << " GFLOPS" << std::endl;

        std::cout << "--> Gap to Baseline:  " << t_tiled/t_base << "x slower than FP32 OneDNN" << std::endl;

        if (verbose) {
            float max_err = 0;
            float* c1 = reinterpret_cast<float*>(C_tiled.raw_data);
            float* c2 = reinterpret_cast<float*>(C_onednn_fp32.raw_data);
            for(int i=0; i<M*B_ref.cols; ++i) {
                max_err = std::max(max_err, std::abs(c1[i] - c2[i]));
            }
            std::cout << "Max Diff (Tiled vs OneDNN): " << max_err << std::endl;
        }
    }

    // Tiled vs Naive comparison
    if (run_naive) {
        MatrixFP32 C_naive(M, B_ref.cols);

        double t_naive = benchmark_gemm(gemm_bf16_naive, A_bf16, B_bf16, C_naive);
        std::cout << "BF16 Naive:             " << t_naive << "s | " << ops/t_naive << " GFLOPS" << std::endl;
        std::cout << "--> Speedup vs Naive: " << t_naive/t_tiled << "x" << std::endl;

        if (verbose) {
            float max_err = 0;
            float* c1 = reinterpret_cast<float*>(C_naive.raw_data);
            float* c2 = reinterpret_cast<float*>(C_tiled.raw_data);
            for(int i=0; i<M*B_ref.cols; ++i) {
                max_err = std::max(max_err, std::abs(c1[i] - c2[i]));
            }
            std::cout << "Max Diff (Naive vs Tiled): " << max_err << std::endl;
        }
    }
}

void run_bench_gemm_int8(int M, const MatrixFP32& B_ref) {
    std::cout << "\n=== Benchmarking INT8 ===" << std::endl;
    MatrixFP32 A_fp32 = generate_random_fp32(M, B_ref.rows);
    MatrixINT8 A_int8(A_fp32);
    MatrixINT8 B_int8(B_ref);
    MatrixINT32 C_naive(M, B_ref.cols);
    MatrixINT32 C_onednn(M, B_ref.cols);
    MatrixINT32 C_tiled(M, B_ref.cols);

    double ops = 2.0 * M * B_ref.cols * B_ref.rows * 1e-9;

    double t_naive = benchmark_gemm(gemm_int8_naive, A_int8, B_int8, C_naive);
    std::cout << "INT8 Naive:  " << t_naive << "s | " << ops/t_naive << " GOPS" << std::endl;

    double t_tiled = benchmark_gemm(gemm_int8_tiled, A_int8, B_int8, C_tiled);
    std::cout << "INT8 Tiled:  " << t_tiled << "s | " << ops/t_tiled << " GOPS" << std::endl;

    try {
        double t_dnnl = benchmark_gemm(gemm_onednn, A_int8, B_int8, C_onednn);
        std::cout << "INT8 OneDNN: " << t_dnnl << "s | " << ops/t_dnnl << " GOPS" << std::endl;
        
        int32_t max_err = 0;
        int32_t* c1 = reinterpret_cast<int32_t*>(C_tiled.raw_data);
        int32_t* c2 = reinterpret_cast<int32_t*>(C_onednn.raw_data);
        for(int i=0; i<M*B_ref.cols; ++i) {
            max_err = std::max(max_err, std::abs(c1[i] - c2[i]));
        }
        std::cout << "Max Diff (Tiled vs OneDNN): " << max_err << std::endl;
    } catch (const dnnl::error& e) {
        std::cout << "INT8 OneDNN skipped: " << e.message << std::endl;
    }
}

int main() {
    // const std::string weight_file = "matrices_data/bert_query.bin";

    std::cout << "omp max threads: " << omp_get_max_threads() << std::endl;
    
    // std::cout << "Loading Base Weight Matrix..." << std::endl;
    // MatrixFP32 B_ref = load_matrix_fp32(weight_file);
    // std::cout << "Loaded " << B_ref.rows << "x" << B_ref.cols << std::endl;

    int M = 32;
    int N = 768;
    int K = 768;
    for (int M: {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024}) {
        std::cout << "\nM=" << M << " K=" << K << " N=" << N << std::endl;
        MatrixFP32 B_ref = generate_random_fp32(K, N);
        run_bench_gemm_bf16(M, B_ref, false, true, false);
    }

    // run_bench_gemm_int8(M, B_ref);

    return 0;
}

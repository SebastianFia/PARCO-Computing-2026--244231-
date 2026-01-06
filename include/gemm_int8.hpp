#pragma once

#include <immintrin.h>

#include "matrix.hpp"
#include "utils.hpp"

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

inline void gemm_int8_naive(const Matrix& A_base, const Matrix& B_base, Matrix& C_base) {
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

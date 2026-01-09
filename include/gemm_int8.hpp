#pragma once

#include <iostream>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <omp.h>
#include <immintrin.h>

#include "matrix.hpp"
#include "utils.hpp"

inline void gemm_int8_tiled(const Matrix& A_base, const Matrix& B_base, Matrix& C_base) {
    const auto& A = static_cast<const MatrixINT8&>(A_base);
    const auto& B = static_cast<const MatrixINT8&>(B_base);
    auto& C = static_cast<MatrixINT32&>(C_base);

    const int M = A.rows;
    const int K = A.cols;
    const int N = B.cols;

    const auto* A_ptr = reinterpret_cast<const int8_t*>(A.raw_data);
    const auto* B_ptr = reinterpret_cast<const int8_t*>(B.raw_data);
    auto* C_ptr = reinterpret_cast<int32_t*>(C.raw_data);

    if (M > N) {
        // return gemm_int8_avx2_tiled_parallel_M(A_ptr, B_ptr, C_ptr, M, N, K);
    } else {
        // return gemm_int8_avx2_tiled_parallel_N(A_ptr, B_ptr, C_ptr, M, N, K);
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

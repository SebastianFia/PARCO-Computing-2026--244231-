#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <functional>
#include <omp.h>

#include "matrix.hpp"
#include "gemm_bf16.hpp"
#include "gemm_int8.hpp"
#include "gemm_onednn.hpp"

constexpr int BENCH_REPEATS = 100;
constexpr int BENCH_WARMUP_REPEATS = 10;

double benchmark_gemm(
    std::function<void(const Matrix&, const Matrix&, Matrix&)> func,
    const Matrix& A, const Matrix& B, Matrix& C
) {
    // Warm-up 
    for (int i = 0; i < BENCH_WARMUP_REPEATS; ++i) {
        func(A, B, C);
    }

    std::vector<double> times(BENCH_REPEATS);
    for (int i = 0; i < BENCH_REPEATS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        times[i] = diff.count();
    }

    std::sort(times.begin(), times.end());
    return (BENCH_REPEATS % 2 == 0) 
        ? 0.5 * (times[BENCH_REPEATS/2 - 1] + times[BENCH_REPEATS/2]) 
        : times[BENCH_REPEATS/2];
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <matrix_name> <dtype: bf16|int8> <impl: ours|onednn> <M> <K> <N>" << std::endl;
        return 1;
    }

    // Parse CLI arguments
    std::string matrix_name = argv[1];
    std::string dtype       = argv[2];
    std::string impl        = argv[3];
    int M = std::stoi(argv[4]);
    int K = std::stoi(argv[5]);
    int N = std::stoi(argv[6]);

    double ops = 2.0 * M * N * K * 1e-9;
    double median_time = 0.0;
    int num_threads = omp_get_max_threads();

    if (dtype == "bf16") {
        MatrixFP32 A_fp32 = generate_random_fp32(M, K);
        MatrixFP32 B_fp32 = generate_random_fp32(K, N); // Reference weights
        MatrixBF16 A_bf16(A_fp32);
        MatrixBF16 B_bf16(B_fp32);
        MatrixFP32 C(M, N);

        if (impl == "ours") {
            median_time = benchmark_gemm(gemm_bf16_tiled, A_bf16, B_bf16, C);
        } else {
            median_time = benchmark_gemm(gemm_onednn_fp32, A_fp32, B_fp32, C);
        }
    } 
    else if (dtype == "int8") {
        MatrixFP32 A_fp32 = generate_random_fp32(M, K);
        MatrixFP32 B_fp32 = generate_random_fp32(K, N);
        MatrixINT8 A_int8(A_fp32);
        MatrixINT8 B_int8(B_fp32);
        MatrixINT32 C(M, N);

        if (impl == "ours") {
            median_time = benchmark_gemm(gemm_int8_tiled, A_int8, B_int8, C);
        } else {
            median_time = benchmark_gemm(gemm_onednn_s8s8s32, A_int8, B_int8, C);
        }
    }

    double throughput = ops / median_time;

    // CSV format: matrix_name,dtype,impl,M,K,N,threads,median_time,throughput
    // Facilitates easy plotting in Python scripts
    std::cout << matrix_name << "," 
              << dtype << "," 
              << impl << "," 
              << M << "," << K << "," << N << "," 
              << num_threads << "," 
              << median_time << "," 
              << throughput << std::endl;

    return 0;
}
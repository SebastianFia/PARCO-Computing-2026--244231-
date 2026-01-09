#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>
#include <omp.h>

#include "oneapi/dnnl/dnnl.hpp"

#include "matrix.hpp"
#include "gemm_bf16.hpp"
#include "gemm_onednn.hpp"

constexpr int BENCH_REPEATS = 100;
constexpr int BENCH_WARMUP_REPEATS = 10;

double benchmark_gemm(
    std::function<void(const Matrix&, const Matrix&, Matrix&)> func,
    const Matrix& A, const Matrix& B, Matrix& C
) {
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
    int mid = BENCH_REPEATS / 2;

    if (BENCH_REPEATS % 2 == 0) {
        return 0.5 * (times[mid - 1] + times[mid]);
    } else {
        return times[mid];
    }
}

int warmup_benchmarks(int M, int K, int N) {
    MatrixFP32 A_fp32 = generate_random_fp32(M, K);
    MatrixFP32 B_fp32 = generate_random_fp32(K, N);
    
    MatrixBF16 A_bf16(A_fp32);
    MatrixBF16 B_bf16(B_fp32);
    
    MatrixFP32 C_ours(M, N);
    MatrixFP32 C_onednn(M, N);

    double t_ours = benchmark_gemm(gemm_bf16_tiled, A_bf16, B_bf16, C_ours);

    double t_onednn = benchmark_gemm(gemm_onednn_fp32, A_fp32, B_fp32, C_onednn);
}

int main() {
    // Configuration
    const int N = 768;
    const std::string filename = "gemm_bench_results.csv";
    
    std::ofstream csv_file(filename);
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return 1;
    }

    // Write CSV Header
    csv_file << "M,K,N,time_ours,time_onednn" << std::endl;

    std::cout << "Starting Sweep (M and K: powers of 2 from 1 to 1024)..." << std::endl;
    std::cout << "Writing results to: " << filename << std::endl;


    // Checksum to prevent dead code optimization
    float checksum = 0;

    // Run the benchmarks once for warmup
    warmup_benchmarks(512, 512, N);

    // Outer loop for M (powers of 2)
    for (int M = 1; M <= 1024; M *= 2) {
        // Inner loop for K (powers of 2)
        for (int K = 1; K <= 1024; K *= 2) {
            MatrixFP32 A_fp32 = generate_random_fp32(M, K);
            MatrixFP32 B_fp32 = generate_random_fp32(K, N);
            
            MatrixBF16 A_bf16(A_fp32);
            MatrixBF16 B_bf16(B_fp32);
            
            MatrixFP32 C_ours(M, N);
            MatrixFP32 C_onednn(M, N);

            double t_ours = benchmark_gemm(gemm_bf16_tiled, A_bf16, B_bf16, C_ours);

            double t_onednn = benchmark_gemm(gemm_onednn_fp32, A_fp32, B_fp32, C_onednn);

            checksum += C_ours.raw_data[0];
            checksum += C_onednn.raw_data[0];

            // Log to Console and File
            std::cout << "M=" << M << "\tK=" << K << "\t| Ours: " << t_ours << "s\toneDNN: " << t_onednn << "s" << std::endl;
            
            csv_file << M << "," 
                     << K << "," 
                     << N << "," 
                     << t_ours << "," 
                     << t_onednn << "\n";
        }
    }

    std::cout << "checksum = " << checksum << std::endl;

    csv_file.close();
    std::cout << "\nBenchmarking complete. Results saved to " << filename << std::endl;

    return 0;
}
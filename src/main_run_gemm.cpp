#include <iostream>
#include <fstream>
#include <chrono>
#include <cassert>
#include <omp.h>

#include "oneapi/dnnl/dnnl.hpp"

#include "matrix.hpp"
#include "gemm_bf16.hpp"
#include "gemm_int8.hpp"
#include "gemm_onednn.hpp"

constexpr int BENCH_REPEATS = 100;
constexpr int BENCH_WARMUP_REPEATS = 10;

// double benchmark_gemm(
//     std::function<void(const Matrix&, const Matrix&, Matrix&)> func, 
//     const Matrix& A, const Matrix& B, Matrix& C
// ) {
//     // Warm-up (discarded)
//     for (int i = 0; i < BENCH_WARMUP_REPEATS; ++i) {
//         func(A, B, C); // Warmup
//     }
//     auto start = std::chrono::high_resolution_clock::now();
//     for(int i = 0; i < BENCH_REPEATS; ++i) {
//         func(A, B, C);
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> diff = end - start;
//     return diff.count() / BENCH_REPEATS;
// }

double benchmark_gemm (
    std::function<void(const Matrix&, const Matrix&, Matrix&)> func,
    const Matrix& A, const Matrix& B, Matrix& C
) {
    // Warm-up (discarded)
    for (int i = 0; i < BENCH_WARMUP_REPEATS; ++i) {
        func(A, B, C);
    }

    // Timed runs 
    std::vector<double> times(BENCH_REPEATS);
    for (int i = 0; i < BENCH_REPEATS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;
        times[i] = diff.count();
    }

    // Compute median
    std::sort(times.begin(), times.end());
    int mid = BENCH_REPEATS / 2;

    if (BENCH_REPEATS % 2 == 0) {
        return 0.5 * (times[mid - 1] + times[mid]);
    } else {
        return times[mid];
    }
}

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
        double t_base = benchmark_gemm(gemm_onednn_fp32, A_fp32, B_ref, C_onednn_fp32);
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

// // Benchmarks our tiled int8 gemm implementation (optionally compares to Baseline and/or naive implementation)
// void run_bench_gemm_int8(int M, const MatrixFP32& B_ref, bool run_naive = false, bool run_baseline = false, bool verbose = false) {
//     MatrixFP32 A_fp32 = generate_random_fp32(M, B_ref.rows);
//     MatrixINT8 A_int8(A_fp32);
//     MatrixINT8 B_int8(B_ref);
//     MatrixINT32 C_tiled(M, B_ref.cols);

//     double ops = 2.0 * M * B_ref.cols * B_ref.rows * 1e-9;

//     double t_tiled = benchmark_gemm(gemm_int8_tiled, A_int8, B_int8, C_tiled);
//     std::cout << "INT8 Tiled:  " << t_tiled << "s | " << ops/t_tiled << " GOPS" << std::endl;

//     if (run_naive) {
//         MatrixINT32 C_naive(M, B_ref.cols);
//         double t_naive = benchmark_gemm(gemm_int8_naive, A_int8, B_int8, C_naive);
//         std::cout << "INT8 Naive:  " << t_naive << "s | " << ops/t_naive << " GOPS" << std::endl;

//         if (verbose) {
//             int32_t max_err = 0;
//             int32_t* c1 = reinterpret_cast<int32_t*>(C_tiled.raw_data);
//             int32_t* c2 = reinterpret_cast<int32_t*>(C_naive.raw_data);

//             for(int i=0; i<M*B_ref.cols; ++i) {
//                 max_err = std::max(max_err, std::abs(c1[i] - c2[i]));
//             }

//             std::cout << "Max Diff (Tiled vs Naive): " << max_err << std::endl;
//         }
//     }

//     if (run_baseline) {
//         try {
//             MatrixINT32 C_onednn(M, B_ref.cols);
//             double t_dnnl = benchmark_gemm(gemm_onednn, A_int8, B_int8, C_onednn);
//             std::cout << "INT8 OneDNN: " << t_dnnl << "s | " << ops/t_dnnl << " GOPS" << std::endl;
            
//             if (verbose) {
//                 int32_t max_err = 0;
//                 int32_t* c1 = reinterpret_cast<int32_t*>(C_tiled.raw_data);
//                 int32_t* c2 = reinterpret_cast<int32_t*>(C_onednn.raw_data);

//                 for(int i=0; i<M*B_ref.cols; ++i) {
//                     max_err = std::max(max_err, std::abs(c1[i] - c2[i]));
//                 }
//                 std::cout << "Max Diff (Tiled vs OneDNN): " << max_err << std::endl;
//             }
//         } catch (const dnnl::error& e) {
//             std::cout << "INT8 OneDNN skipped: " << e.message << std::endl;
//         }
//     }
// }

int main() {
    std::cout << "omp max threads: " << omp_get_max_threads() << std::endl;
    
    // std::cout << "Loading Base Weight Matrix..." << std::endl;
    // MatrixFP32 B_ref = load_matrix_fp32(weight_file);
    // std::cout << "Loaded " << B_ref.rows << "x" << B_ref.cols << std::endl;

    int N = 768;
    int K = 768;
    MatrixFP32 B_ref = generate_random_fp32(K, N);

    // for (int M: {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024}) {
    // for (int M: {128, 128, 128, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}) {
    for (int M: {128, 128, 256, 512}) {
        std::cout << "\nM=" << M << " K=" << K << " N=" << N << std::endl;
        run_bench_gemm_bf16(M, B_ref, false, true, false);
        // run_bench_gemm_int8(M, B_ref, false, true, false);
    }

    return 0;
}

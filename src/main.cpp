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

// Include OneDNN
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

constexpr size_t ALIGNMENT = 64;

// --- 0. HELPER TYPES & BF16 UTILS ---

// Bfloat16 is just uint16_t in storage
using bfloat16_t = uint16_t;

// Helper: Float32 <-> BF16 (Round-to-Nearest-Even)
bfloat16_t fp32_to_bf16(float f) {
    uint32_t input_bits = *reinterpret_cast<uint32_t*>(&f);
    // Add 0x7FFF + LSB of the top 16 bits to handle rounding
    // This is a standard software approximation for RNE
    uint32_t least_significant_bit = (input_bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + least_significant_bit;
    input_bits += rounding_bias;
    return static_cast<bfloat16_t>(input_bits >> 16);
}

inline float bf16_to_fp32(bfloat16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    return *reinterpret_cast<float*>(&bits);
}

// --- 1. MATRIX CLASS HIERARCHY ---

enum class DType { FP32, BF16, INT8, INT32 };

struct Matrix {
    int rows;
    int cols;
    size_t dtype_size;
    DType dtype;
    char* raw_data; // Raw bytes storage

    Matrix(int r, int c, size_t d_sz, DType d_type) 
        : rows(r), cols(c), dtype_size(d_sz), dtype(d_type) {
        size_t bytes = rows * cols * dtype_size;
        raw_data = static_cast<char*>(std::aligned_alloc(ALIGNMENT, bytes));
        if (!raw_data) throw std::runtime_error("Allocation failed");
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
    
    // Construct from FP32
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

    // Construct from FP32 with simple Min-Max Quantization
    MatrixINT8(const MatrixFP32& src) : Matrix(src.rows, src.cols, sizeof(int8_t), DType::INT8) {
        const float* src_ptr = reinterpret_cast<const float*>(src.raw_data);
        int num_elements = rows * cols;

        // 1. Find Min/Max
        float min_val = src_ptr[0];
        float max_val = src_ptr[0];
        for(int i=1; i<num_elements; ++i) {
            if(src_ptr[i] < min_val) min_val = src_ptr[i];
            if(src_ptr[i] > max_val) max_val = src_ptr[i];
        }

        // 2. Compute Scale & ZP (Mapping min->-128, max->127)
        // Range of int8 is [-128, 127] -> 255 steps
        scale = (max_val - min_val) / 255.0f;
        if (scale == 0.0f) scale = 1.0f; // Prevent div by zero for constant matrix
        
        // zero_point = -round(min / scale) - 128
        // We shift to ensure min maps to -128
        zero_point = static_cast<int32_t>(std::round(-min_val / scale)) - 128;

        // 3. Quantize
        int8_t* dst_ptr = reinterpret_cast<int8_t*>(raw_data);
        for(int i=0; i<num_elements; ++i) {
            float val = src_ptr[i];
            int32_t q = static_cast<int32_t>(std::round(val / scale)) + zero_point;
            // Clamp to int8 range
            q = std::max(-128, std::min(127, q));
            dst_ptr[i] = static_cast<int8_t>(q);
        }
    }
};

// Helper struct for Integer GEMM output (Accumulators are usually 32-bit)
struct MatrixINT32 : public Matrix {
    MatrixINT32(int r, int c) : Matrix(r, c, sizeof(int32_t), DType::INT32) {}
    
    int32_t& at(int i, int j) {
        return reinterpret_cast<int32_t*>(raw_data)[i * cols + j];
    }
};

// --- 2. HELPERS: IO & GENERATION ---

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
    // Using a smaller range for inputs to avoid overflow in naive INT8 accumulators quickly
    std::normal_distribution<float> d(0.0f, 1.0f); 
    float* ptr = reinterpret_cast<float*>(mat.raw_data);
    for (int i = 0; i < rows * cols; ++i) ptr[i] = d(gen);
    return mat;
}

// --- 3. KERNELS (NAIVE & ONEDNN) ---

// --- FP32 ---
void gemm_fp32_naive(const Matrix& A_base, const Matrix& B_base, Matrix& C_base) {
    const auto& A = static_cast<const MatrixFP32&>(A_base);
    const auto& B = static_cast<const MatrixFP32&>(B_base);
    auto& C = static_cast<MatrixFP32&>(C_base);

    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k) {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = sum;
        }
    }
}

// --- BF16 ---
void gemm_bf16_naive(const Matrix& A_base, const Matrix& B_base, Matrix& C_base) {
    const auto& A = static_cast<const MatrixBF16&>(A_base);
    const auto& B = static_cast<const MatrixBF16&>(B_base);
    auto& C = static_cast<MatrixFP32&>(C_base); // Output is FP32 (standard for BF16 GEMM)

    const bfloat16_t* A_ptr = reinterpret_cast<const bfloat16_t*>(A.raw_data);
    const bfloat16_t* B_ptr = reinterpret_cast<const bfloat16_t*>(B.raw_data);
    float* C_ptr = reinterpret_cast<float*>(C.raw_data);

    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k) {
                // Upcast BF16 -> FP32 for computation
                float a_val = bf16_to_fp32(A_ptr[i * A.cols + k]);
                float b_val = bf16_to_fp32(B_ptr[k * B.cols + j]);
                sum += a_val * b_val;
            }
            C_ptr[i * C.cols + j] = sum;
        }
    }
}

// --- INT8 ---
void gemm_int8_naive(const Matrix& A_base, const Matrix& B_base, Matrix& C_base) {
    const auto& A = static_cast<const MatrixINT8&>(A_base);
    const auto& B = static_cast<const MatrixINT8&>(B_base);
    // Output for INT8 GEMM is typically INT32 accumulators
    auto& C = static_cast<MatrixINT32&>(C_base); 

    const int8_t* A_ptr = reinterpret_cast<const int8_t*>(A.raw_data);
    const int8_t* B_ptr = reinterpret_cast<const int8_t*>(B.raw_data);
    int32_t* C_ptr = reinterpret_cast<int32_t*>(C.raw_data);

    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            int32_t sum = 0; // 32-bit accumulator
            for (int k = 0; k < A.cols; ++k) {
                // Integer Multiply-Add
                sum += static_cast<int32_t>(A_ptr[i * A.cols + k]) * static_cast<int32_t>(B_ptr[k * B.cols + j]);
            }
            C_ptr[i * C.cols + j] = sum;
        }
    }
}

// --- ONEDNN GENERIC WRAPPER ---
void gemm_onednn(const Matrix& A, const Matrix& B, Matrix& C) {
    engine eng(engine::kind::cpu, 0);
    stream engine_stream(eng);

    memory::dims a_dims = {A.rows, A.cols};
    memory::dims b_dims = {B.rows, B.cols};
    memory::dims c_dims = {C.rows, C.cols};

    // Map internal DType to OneDNN memory data_type
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

// --- 4. BENCHMARKING ENGINE ---

// Returns average time in seconds
double benchmark_gemm(std::function<void(const Matrix&, const Matrix&, Matrix&)> func, 
                      int repeats, 
                      const Matrix& A, const Matrix& B, Matrix& C) {
    // Warmup
    func(A, B, C);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<repeats; ++i) {
        func(A, B, C);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    return diff.count() / repeats;
}

// --- 5. RUNNERS PER DTYPE ---

void run_bench_gemm_fp32(int M, const MatrixFP32& B_ref) {
    std::cout << std::endl << "=== Benchmarking FP32 ===" << std::endl;
    // 1. Setup Inputs
    MatrixFP32 A = generate_random_fp32(M, B_ref.rows);
    MatrixFP32 C_naive(M, B_ref.cols);
    MatrixFP32 C_onednn(M, B_ref.cols);

    double ops = 2.0 * M * B_ref.cols * B_ref.rows * 1e-9;
    int repeats = 5;

    // 2. Run Naive
    double t_naive = benchmark_gemm(gemm_fp32_naive, repeats, A, B_ref, C_naive);
    std::cout << "FP32 Naive:  " << t_naive << "s | " << ops/t_naive << " GFLOPS" << std::endl;

    // 3. Run OneDNN
    double t_dnnl = benchmark_gemm(gemm_onednn, repeats, A, B_ref, C_onednn);
    std::cout << "FP32 OneDNN: " << t_dnnl << "s | " << ops/t_dnnl << " GFLOPS" << std::endl;

    // 4. Correctness
    float max_err = 0.0f;
    for(int i=0; i<M*B_ref.cols; ++i) {
        max_err = std::max(max_err, std::abs(C_naive.at(0, i) - C_onednn.at(0, i)));
    }
    std::cout << "Max Diff: " << max_err << std::endl;
}

void run_bench_gemm_bf16(int M, const MatrixFP32& B_ref) {
    std::cout << std::endl << "=== Benchmarking BF16 ===" << std::endl;
    // 1. Setup Inputs & Convert
    MatrixFP32 A_fp32 = generate_random_fp32(M, B_ref.rows);
    
    MatrixBF16 A_bf16(A_fp32);
    MatrixBF16 B_bf16(B_ref);
    
    // Output of BF16 GEMM is typically FP32
    MatrixFP32 C_naive(M, B_ref.cols);
    MatrixFP32 C_onednn(M, B_ref.cols);

    double ops = 2.0 * M * B_ref.cols * B_ref.rows * 1e-9;
    int repeats = 5;

    // 2. Run Naive (BF16 input -> upcast -> FP32 output)
    double t_naive = benchmark_gemm(gemm_bf16_naive, repeats, A_bf16, B_bf16, C_naive);
    std::cout << "BF16 Naive:  " << t_naive << "s | " << ops/t_naive << " GFLOPS" << std::endl;

    // 3. Run OneDNN
    try {
        double t_dnnl = benchmark_gemm(gemm_onednn, repeats, A_bf16, B_bf16, C_onednn);
        std::cout << "BF16 OneDNN: " << t_dnnl << "s | " << ops/t_dnnl << " GFLOPS" << std::endl;

        // 4. Correctness
        float max_err = 0.0f;
        float* c1 = reinterpret_cast<float*>(C_naive.raw_data);
        float* c2 = reinterpret_cast<float*>(C_onednn.raw_data);
        for(int i=0; i<M*B_ref.cols; ++i) {
            max_err = std::max(max_err, std::abs(c1[i] - c2[i]));
        }
        std::cout << "Max Diff: " << max_err << std::endl;

    } catch (const dnnl::error& e) {
        std::cout << "BF16 OneDNN skipped: " << e.message << " (Likely no hardware support)" << std::endl;
    }
}

void run_bench_gemm_int8(int M, const MatrixFP32& B_ref) {
    std::cout << std::endl << "=== Benchmarking INT8 ===" << std::endl;
    // 1. Setup Inputs & Quantize
    MatrixFP32 A_fp32 = generate_random_fp32(M, B_ref.rows);
    
    MatrixINT8 A_int8(A_fp32);
    MatrixINT8 B_int8(B_ref);
    
    // Output of INT8 GEMM is typically INT32
    MatrixINT32 C_naive(M, B_ref.cols);
    MatrixINT32 C_onednn(M, B_ref.cols);

    // OPS for Int8 is usually counted as GOPs
    double ops = 2.0 * M * B_ref.cols * B_ref.rows * 1e-9;
    int repeats = 5;

    // 2. Run Naive (INT8 input -> INT32 output)
    double t_naive = benchmark_gemm(gemm_int8_naive, repeats, A_int8, B_int8, C_naive);
    std::cout << "INT8 Naive:  " << t_naive << "s | " << ops/t_naive << " GOPS" << std::endl;

    // 3. Run OneDNN
    // Note: OneDNN s8*s8 -> s32 is standard
    try {
        double t_dnnl = benchmark_gemm(gemm_onednn, repeats, A_int8, B_int8, C_onednn);
        std::cout << "INT8 OneDNN: " << t_dnnl << "s | " << ops/t_dnnl << " GOPS" << std::endl;

        // 4. Correctness
        int32_t max_err = 0;
        int32_t* c1 = reinterpret_cast<int32_t*>(C_naive.raw_data);
        int32_t* c2 = reinterpret_cast<int32_t*>(C_onednn.raw_data);
        for(int i=0; i<M*B_ref.cols; ++i) {
            max_err = std::max(max_err, std::abs(c1[i] - c2[i]));
        }
        std::cout << "Max Diff (Int32): " << max_err << std::endl;

    } catch (const dnnl::error& e) {
        std::cout << "INT8 OneDNN skipped: " << e.message << std::endl;
    }
}

// --- 6. MAIN ---

int main() {
    // Ensure you have matrices_data/bert_query.bin 
    // (Generated by the python script)
    const std::string weight_file = "matrices_data/bert_query.bin";
    
    std::cout << "Loading Base Weight Matrix..." << std::endl;
    MatrixFP32 B_ref = load_matrix_fp32(weight_file);
    std::cout << "Loaded " << B_ref.rows << "x" << B_ref.cols << std::endl;

    int M = 64; // Batch Size

    // Run Benchmarks

    for (int M: {32, 64, 128, 256}) {
        std::cout << std::endl << "=== Bench with M=" << M << " ===" << std::endl;
        run_bench_gemm_fp32(M, B_ref);
        run_bench_gemm_bf16(M, B_ref);
        // run_bench_gemm_int8(M, B_ref);
    }

    return 0;
}
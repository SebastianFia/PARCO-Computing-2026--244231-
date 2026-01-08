#pragma once

#include <iostream>
#include <fstream>
#include <random>
#include <cassert>
#include <memory>
#include <cstring>

#include "bfloat16.hpp"
#include "utils.hpp"

constexpr size_t ALIGNMENT = 64; // For avx2 and cache line alignment

enum class DType { FP32, BF16, INT8, INT32 };

struct Matrix {
    int rows;
    int cols;
    size_t dtype_size;
    DType dtype;
    char* raw_data; 

    Matrix(int r, int c, size_t d_sz, DType d_type) 
        : rows(r), cols(c), dtype_size(d_sz), dtype(d_type) {
        size_t bytes = round_up(rows * cols * dtype_size, ALIGNMENT);
        raw_data = static_cast<char*>(aligned_alloc(ALIGNMENT, bytes));
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

inline MatrixFP32 load_matrix_fp32(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) { std::cerr << "Error: " << filename << std::endl; exit(1); }
    int rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    MatrixFP32 mat(rows, cols);
    file.read(mat.raw_data, rows * cols * sizeof(float));
    return mat;
}

inline MatrixFP32 generate_random_fp32(int rows, int cols) {
    MatrixFP32 mat(rows, cols);
    std::mt19937 gen(42);
    std::normal_distribution<float> d(0.0f, 1.0f); 
    float* ptr = reinterpret_cast<float*>(mat.raw_data);
    for (int i = 0; i < rows * cols; ++i) ptr[i] = d(gen);
    return mat;
}

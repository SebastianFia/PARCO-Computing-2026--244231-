#pragma once

#include <cassert>
#include <iostream>

#include "oneapi/dnnl/dnnl.hpp"

#include "matrix.hpp"

using namespace dnnl;

inline dnnl::memory wrap_memory(const Matrix& m, const memory::desc& md, const engine& eng) {
    return memory(md, eng, m.raw_data);
}

inline void gemm_onednn_s8s8s32(const Matrix& A, const Matrix& B, Matrix& C) {
    assert(A.dtype == DType::INT8 && "Matrix A must be INT8");
    assert(B.dtype == DType::INT8 && "Matrix B must be INT8");
    assert(C.dtype == DType::INT32 && "Matrix C must be INT32");
    assert(A.cols == B.rows && "Dimension mismatch: A.cols != B.rows");
    assert(C.rows == A.rows && C.cols == B.cols && "Output dimension mismatch");

    memory::dim M = A.rows;
    memory::dim K = A.cols;
    memory::dim N = B.cols;

    static engine eng(engine::kind::cpu, 0);
    static stream s(eng);

    auto a_md = memory::desc({M, K}, memory::data_type::s8, memory::format_tag::ab);
    auto b_md = memory::desc({K, N}, memory::data_type::s8, memory::format_tag::ab);
    auto c_md = memory::desc({M, N}, memory::data_type::s32, memory::format_tag::ab);

    auto matmul_d = matmul::desc(a_md, b_md, c_md);
    auto matmul_pd = matmul::primitive_desc(matmul_d, eng);
    auto matmul_prim = matmul(matmul_pd);

    auto a_mem = wrap_memory(A, a_md, eng);
    auto b_mem = wrap_memory(B, b_md, eng);
    auto c_mem = wrap_memory(C, c_md, eng);

    matmul_prim.execute(s, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem}
    });
    s.wait();
}

inline void gemm_onednn_fp32(const Matrix& A, const Matrix& B, Matrix& C) {
    assert(A.dtype == DType::FP32 && "Matrix A must be FP32");
    assert(B.dtype == DType::FP32 && "Matrix B must be FP32");
    assert(C.dtype == DType::FP32 && "Matrix C must be FP32");
    assert(A.cols == B.rows && "Dimension mismatch: A.cols != B.rows");
    assert(C.rows == A.rows && C.cols == B.cols && "Output dimension mismatch");

    memory::dim M = A.rows;
    memory::dim K = A.cols;
    memory::dim N = B.cols;

    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    auto a_md = memory::desc({M, K}, memory::data_type::f32, memory::format_tag::ab);
    auto b_md = memory::desc({K, N}, memory::data_type::f32, memory::format_tag::ab);
    auto c_md = memory::desc({M, N}, memory::data_type::f32, memory::format_tag::ab);

    auto matmul_d = matmul::desc(a_md, b_md, c_md);
    auto matmul_pd = matmul::primitive_desc(matmul_d, eng);
    auto matmul_prim = matmul(matmul_pd);

    auto a_mem = wrap_memory(A, a_md, eng);
    auto b_mem = wrap_memory(B, b_md, eng);
    auto c_mem = wrap_memory(C, c_md, eng);

    matmul_prim.execute(s, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem}
    });
    s.wait();
}
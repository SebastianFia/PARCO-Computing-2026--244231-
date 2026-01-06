#pragma once

#include "oneapi/dnnl/dnnl.hpp"

#include "matrix.hpp"

using namespace dnnl;

inline void gemm_onednn(const Matrix& A, const Matrix& B, Matrix& C) {
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


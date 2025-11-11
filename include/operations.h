#ifndef __OPERATIONS_H__
#define __OPERATIONS_H__

#include "vector.h"
#include "matrix.h"
#include "matrix_coo.h"
#include "matrix_csr.h"
#include "matrix_csb.h"

typedef struct mat_vec_mul_args_t {
    const matrix_t* m;
    const vector_t* x;
    vector_t* y;
} mat_vec_mul_args_t;
void mat_vec_mul(const matrix_t* m, const vector_t* x, vector_t* y);
void* mat_vec_mul_wrapper(void* args);
void mat_vec_mul_omp(const matrix_t* m, const vector_t* x, vector_t* y);
void* mat_vec_mul_omp_wrapper(void* args);

typedef struct spmv_coo_args_t {
    const matrix_coo_t* m;
    const vector_t* x;
    vector_t* y;
} spmv_coo_args_t;
void spmv_coo(const matrix_coo_t* m, const vector_t* x, vector_t* y);
void* spmv_coo_wrapper(void* args);
void spmv_coo_omp(const matrix_coo_t* m, const vector_t* x, vector_t* y);
void* spmv_coo_omp_wrapper(void* args);

typedef struct spmv_csr_args_t {
    const matrix_csr_t* m;
    const vector_t* x;
    vector_t* y;
} spmv_csr_args_t;
void spmv_csr(const matrix_csr_t* m, const vector_t* x, vector_t* y);
void* spmv_csr_wrapper(void* args);
void spmv_csr_omp(const matrix_csr_t* m, const vector_t* x, vector_t* y);
void* spmv_csr_omp_wrapper(void* args);

typedef struct spmv_csb_args_t {
    const matrix_csb_t* m;
    const vector_t* x;
    vector_t* y;
} spmv_csb_args_t;
void spmv_csb(const matrix_csb_t* m, const vector_t* x, vector_t* y);
void* spmv_csb_wrapper(void* args);
void spmv_csb_omp(const matrix_csb_t* m, const vector_t* x, vector_t* y);
void* spmv_csb_omp_wrapper(void* args);

#endif 

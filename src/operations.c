#include <stdlib.h>
#include <omp.h>
#include <stdio.h>

#include "vector.h"
#include "matrix.h"
#include "matrix_coo.h"
#include "matrix_csr.h"
#include "operations.h"

void mat_vec_mul(const matrix_t *m, const vector_t *x, vector_t *y)
{
    const int nrow = m->nrow;
    const int ncol = m->ncol;
    const double *mval = m->val;
    const double *xval = x->val;
    double *yval = y->val;

    for (int i = 0; i < nrow; ++i)
    {
        double sum = 0;
        for (int j = 0; j < ncol; ++j)
        {
            sum += mval[i * ncol + j] * xval[j];
        }
        yval[i] = sum;
    }
}

void *mat_vec_mul_wrapper(void *args)
{
    mat_vec_mul_args_t *args_ = (mat_vec_mul_args_t *)args;
    mat_vec_mul(args_->m, args_->x, args_->y);
    return NULL;
}

void mat_vec_mul_omp(const matrix_t *m, const vector_t *x, vector_t *y)
{
    const int nrow = m->nrow;
    const int ncol = m->ncol;
    const double *mval = m->val;
    const double *xval = x->val;
    double *yval = y->val;

    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < nrow; ++i)
    {
        double sum = 0;
        for (int j = 0; j < ncol; ++j)
            sum += mval[i * ncol + j] * xval[j];
        yval[i] = sum;
    }
}

void *mat_vec_mul_omp_wrapper(void *args)
{
    mat_vec_mul_args_t *args_ = (mat_vec_mul_args_t *)args;
    mat_vec_mul_omp(args_->m, args_->x, args_->y);
    return NULL;
}

void spmv_coo(const matrix_coo_t *m, const vector_t *x, vector_t *y)
{
    int nnz = m->nnz;
    const int *row = m->row;
    const int *col = m->col;
    const double *xval = x->val;
    const double *mval = m->val;
    double *yval = y->val;

    for (int k = 0; k < nnz; ++k)
    {
        int i = row[k];
        int j = col[k];
        yval[i] += mval[k] * xval[j];
    }
}

void *spmv_coo_wrapper(void *args)
{
    spmv_coo_args_t *args_ = (spmv_coo_args_t *)args;
    spmv_coo(args_->m, args_->x, args_->y);
    return NULL;
}

void spmv_coo_omp(const matrix_coo_t *m, const vector_t *x, vector_t *y)
{
    if (omp_get_max_threads() == 1)
        return spmv_coo(m, x, y);

    const int nnz = m->nnz;
    const int nrow = m->nrow;
    const int *row = m->row;
    const int *col = m->col;
    const double *xval = x->val;
    const double *mval = m->val;
    double *yval = y->val;

    #pragma omp parallel
    {
        // We use a local y_private because it vastly improves performance,
        // justifying the use of multiple threads (which would otherwise only
        // slow us down in this version). However, doing so we allocate much
        // more memory: num_threads * nrow * sizeof(double)

        double *y_private = calloc(nrow, sizeof(double));
        if (y_private == NULL)
        {
            fprintf(stderr, "Failed to allocate y_private in spmv_coo_omp\n");
            exit(1);
        }

        // We add nowait so when a thread has finished can directly go to the
        // critical section and start adding to the shared y

        #pragma omp for schedule(runtime) nowait
        for (int k = 0; k < nnz; ++k)
        {
            const int i = row[k];
            const int j = col[k];
            // #pragma omp atomic
            // yval[i] += mval[k] * xval[j];
            y_private[i] += mval[k] * xval[j];
        }

        #pragma omp critical
        {
            for (int i = 0; i < nrow; ++i)
                yval[i] += y_private[i];
        }

        free(y_private);
    }
}

void *spmv_coo_omp_wrapper(void *args)
{
    spmv_coo_args_t *args_ = (spmv_coo_args_t *)args;
    spmv_coo_omp(args_->m, args_->x, args_->y);
    return NULL;
}

void spmv_csr(const matrix_csr_t *m, const vector_t *x, vector_t *y)
{
    const int nrow = m->nrow;
    const int *col = m->col;
    const int *row_ptr = m->row_ptr;
    const double *mval = m->val;
    const double *xval = x->val;
    double *yval = y->val;

    int k = 0;
    for (int i = 0; i < nrow; ++i)
    {
        const int end_k = row_ptr[i + 1];
        double sum = 0;
        for (; k < end_k; ++k)
        {
            int j = col[k];
            sum += mval[k] * xval[j];
        }
        yval[i] = sum;
    }
}

void *spmv_csr_wrapper(void *args)
{
    spmv_csr_args_t *args_ = (spmv_csr_args_t *)args;
    spmv_csr(args_->m, args_->x, args_->y);
    return NULL;
}

void spmv_csr_omp(const matrix_csr_t *m, const vector_t *x, vector_t *y)
{
    if (omp_get_max_threads() == 1)
        return spmv_csr(m, x, y);

    const int nrow = m->nrow;
    const int *col = m->col;
    const int *row_ptr = m->row_ptr;
    const double *mval = m->val;
    const double *xval = x->val;
    double *yval = y->val;

    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < nrow; ++i)
    {
        const int start_k = row_ptr[i];
        const int end_k = row_ptr[i + 1];
        double sum = 0;

        for (int k = start_k; k < end_k; ++k)
        {
            int j = col[k];
            sum += mval[k] * xval[j];
        }
        yval[i] = sum;
    }
}

void *spmv_csr_omp_wrapper(void *args)
{
    spmv_csr_args_t *args_ = (spmv_csr_args_t *)args;
    spmv_csr_omp(args_->m, args_->x, args_->y);
    return NULL;
}

void spmv_csb(const matrix_csb_t *m, const vector_t *x, vector_t *y)
{
    return;
    const uint32_t nblocks_row = m->nblocks_row;
    const uint32_t nblocks_col = m->nblocks_col;
    const uint32_t nrow = m->nrow;
    const uint32_t ncol = m->ncol;
    const uint32_t block_size = m->block_size;
    const uint32_t* rowcol = m->rowcol;
    const uint32_t* blk_ptr = m->blk_ptr;

    // Loop over block rows rr
    for (uint32_t rr = 0; rr < nblocks_row; ++rr){ 
        // Loop over block columns cc
        for (uint32_t cc = 0; cc < nblocks_col; ++cc) {
            const uint32_t b = rr * nblocks_col + cc; // block index
            const uint32_t start_k = blk_ptr[b];
            const uint32_t end_k = blk_ptr[b+1];
            double sum = 0;

            // Loop over nnzs
            for (uint32_t k = start_k; k < end_k; ++k) {
                // Unpack local row and col
                const uint32_t packed_rc = m->rowcol[k];
                const uint32_t local_r = packed_rc << 16; // High 16 bits
                const uint32_t local_c = packed_rc && 0xFFFF; // Low 16 bits

                const uint32_t global_r = rr*block_size + local_r;
                const uint32_t global_c = cc*block_size + local_c;

                y->val[global_r] += m->val[k] * x->val[global_c];
            }
        }
    }
}

void *spmv_csb_wrapper(void *args)
{
    spmv_csb_args_t *args_ = (spmv_csb_args_t *)args;
    spmv_csb(args_->m, args_->x, args_->y);
    return NULL;
}

void spmv_csb_omp(const matrix_csb_t *m, const vector_t *x, vector_t *y)
{

}

void *spmv_csb_omp_wrapper(void *args)
{
    spmv_csb_args_t *args_ = (spmv_csb_args_t *)args;
    spmv_csb_omp(args_->m, args_->x, args_->y);
    return NULL;
}
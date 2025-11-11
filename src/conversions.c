#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "conversions.h"

matrix_coo_t *matrix_coo_from_dense(const matrix_t *dense)
{
    if (dense == NULL || dense->val == NULL)
    {
        fprintf(stderr, "Cannot create matrix from NULL pointers.\n");
        return NULL;
    }

    // Count nonzero elements
    int nrow = dense->nrow;
    int ncol = dense->ncol;
    int nnz = 0;
    for (int i = 0; i < nrow; ++i)
    {
        for (int j = 0; j < ncol; ++j)
        {
            if (dense->val[i * ncol + j] != 0.0)
                ++nnz;
        }
    }

    // Build COO matrix
    matrix_coo_t *coo = malloc(sizeof(*coo));
    if (coo == NULL)
    {
        perror("Failed to allocate coo matrix");
        return NULL;
    }
    coo->nrow = dense->nrow;
    coo->ncol = dense->ncol;
    coo->nnz = nnz;
    coo->row = malloc(sizeof(*coo->row) * nnz);
    coo->col = malloc(sizeof(*coo->col) * nnz);
    coo->val = malloc(sizeof(*coo->val) * nnz);
    if (coo->row == NULL || coo->col == NULL || coo->val == NULL)
    {
        perror("Failed to allocate coo matrix");
        free(coo->row);
        free(coo->col);
        free(coo->val);
        free(coo);
        return NULL;
    }

    // Find each kth nnz element in the dense matrix,
    // and store its val, row and col in the coo matrix at index k
    int k = 0;
    for (int i = 0; i < nrow; ++i)
    {
        for (int j = 0; j < ncol; ++j)
        {
            if (dense->val[i * ncol + j] != 0.0)
            {
                coo->val[k] = dense->val[i * ncol + j];
                coo->row[k] = i;
                coo->col[k] = j;
                ++k;
            }
        }
    }

    return coo;
}

matrix_t *matrix_dense_form_coo(const matrix_coo_t *coo)
{
    if (coo == NULL || coo->col == NULL || coo->row == NULL || coo->val == NULL)
    {
        fprintf(stderr, "Cannot create matrix from NULL pointers.\n");
        return NULL;
    }

    int nrow = coo->nrow;
    int ncol = coo->ncol;
    int nnz = coo->nnz;

    matrix_t *dense = malloc(sizeof(matrix_t));
    if (dense == NULL)
    {
        perror("Failed to allocate dense matrix");
        return NULL;
    }

    dense->nrow = nrow;
    dense->ncol = ncol;
    dense->val = calloc(nrow * ncol, sizeof(double));
    if (dense->val == NULL)
    {
        perror("Failed to allocate dense matrix");
        free(dense);
        return NULL;
    }

    for (int k = 0; k < nnz; ++k)
    {
        int i = coo->row[k];
        int j = coo->col[k];
        dense->val[i * ncol + j] = coo->val[k];
    }

    return dense;
}

matrix_csr_t *matrix_csr_from_coo(const matrix_coo_t *coo)
{
    if (coo == NULL || coo->col == NULL || coo->row == NULL || coo->val == NULL)
    {
        fprintf(stderr, "Cannot create matrix from NULL pointers.\n");
        return NULL;
    }

    // The coo_copy variable is used only if we create a copy of the COO.
    // We free it before returning from the function in every case.
    matrix_coo_t *coo_copy = NULL;

    // If the coo matrix isn't sorted, copy it and sort the copy
    if (!matrix_coo_is_sorted_by_coordinates(coo))
    {
        coo_copy = matrix_coo_copy(coo);
        if (coo_copy == NULL)
        {
            fprintf(stderr, "Failed to create tmp matrix for sorting.\n");
            free(coo_copy);
            return NULL;
        }
        matrix_coo_sort_by_coordinates(coo_copy);
        coo = coo_copy;
    };

    // Count the number of nnz elemts for each ith row,
    // and store it in the compressed_row at location i+1.
    int nrow = coo->nrow;
    int nnz = coo->nnz;
    int *row_ptr = calloc(nrow + 1, sizeof(int));
    if (row_ptr == NULL)
    {
        perror("Failed to allocate row_ptr");
        free(coo_copy);
        return NULL;
    }
    for (int k = 0; k < nnz; ++k)
    {
        int i = coo->row[k];
        row_ptr[i + 1]++;
    }

    // Compute the prefix sum of the compressed row
    for (int i = 0; i < nrow; ++i)
        row_ptr[i + 1] += row_ptr[i];

    // Create the CSR matrix
    matrix_csr_t *csr = malloc(sizeof(matrix_csr_t));

    if (csr == NULL)
    {
        perror("Failed to allocate csr matrix");
        free(coo_copy);
        return NULL;
    }

    csr->nrow = coo->nrow;
    csr->ncol = coo->ncol;
    csr->nnz = coo->nnz;
    csr->row_ptr = row_ptr;
    csr->col = malloc(sizeof(int) * coo->nnz);
    csr->val = malloc(sizeof(double) * coo->nnz);

    if (csr->col == NULL || csr->val == NULL)
    {
        fprintf(stderr, "Failed to allocate csr matrix.\n");
        free(csr);
        free(coo_copy);
        return NULL;
    }

    memcpy((void *)csr->col, (void *)coo->col, sizeof(int) * coo->nnz);
    memcpy((void *)csr->val, (void *)coo->val, sizeof(double) * coo->nnz);

    return csr;
}

matrix_csb_t *matrix_csb_from_coo(const matrix_coo_t *coo)
{
    if (!coo || !coo->row || !coo->col || !coo->val)
    {
        fprintf(stderr, "Got NULL pointers in matrix_csb_from_coo funciton.\n");
        return NULL;
    }

    matrix_csb_t *csb = malloc(sizeof(*csb));
    if (!csb)
    {
        perror("Failed to allocate csb matrix");
        return NULL;
    }
    
    return csb;
}
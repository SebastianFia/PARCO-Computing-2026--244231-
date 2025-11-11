#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <stdlib.h>
#include "utils.h"
#include "matrix_coo.h"

// A matrix of nrows*cols elements, stored in a contiguous array.
typedef struct matrix_t {
    int nrow;
    int ncol;
    double* val;
} matrix_t;

// Creates a dynamically allocated matrix filled with zeros and of shape nrow x ncol.
// Returns a pointer to it, or NULL on failure.
matrix_t* matrix_create_zeros(int nrow, int ncol);

// Creates a dynamically allocated sparse matrix of shape nrow x ncol.
// Returns a pointer to it, or NULL on failure.
// Each entry has a p_nnz probability of being a nonzero value.
// Each nonzero value is uniformly distributed between min and max.
matrix_t* matrix_create_sparse(int nrow, int ncol, double p_nnz, double min, double max);

// Print a matrix.
void matrix_print(const matrix_t* m);

// Frees a dynamically allocated matrix and its contents.
// Does nothing on NULLs.
void matrix_free(matrix_t* m);

#endif

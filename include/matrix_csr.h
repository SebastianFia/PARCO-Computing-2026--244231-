#ifndef __MATRIX_CSR_H__
#define __MATRIX_CSR_H__

#include <stdlib.h>

// A matrix of shape nrows x cols, with nnz nonzero elements stored in 
// Compressed Sparse Row format.
typedef struct matrix_csr_t {
    int nnz; // Number of nonzero elements
    int nrow; // Number of rows
    int ncol; // Number of columns
    int* row_ptr; // Prefix sum of number of nnz per row
    int* col; // Array of column indicies
    double* val; // Array of the values
} matrix_csr_t;

// Print a CSR matrix.
void matrix_csr_print(const matrix_csr_t* m);

// Get the sparsity ratio of the matrix, a number between 0 and 1 computed as:
// number of zero elements over total number of elements.
double matrix_csr_get_sparsity(const matrix_csr_t* m);

int matrix_csr_get_num_empty_rows(const matrix_csr_t* m);

int matrix_csr_get_num_rows_with_one_nonzero(const matrix_csr_t* m);

double matrix_csr_get_average_nonzeros_per_row(const matrix_csr_t* m);

// Print the histogram of the "number of nonzero elements per matrix row".
// Simply put, histogram[i] is "the number of rows with i nonzero elements".
void matrix_csr_print_nnz_per_row_histogram(const matrix_csr_t* m);

// Free a dynamically allocated CSR matrix and its contents.
void matrix_csr_free(matrix_csr_t* m);


double matrix_csr_get_spmv_bytes_moved(const matrix_csr_t* m);

double matrix_csr_get_spmv_total_flops(const matrix_csr_t* m);

#endif

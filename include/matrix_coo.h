#ifndef __MATRIX_COO_H__
#define __MATRIX_COO_H__

#include <stdlib.h>
#include "matrix.h"

// A matrix of shape nrows x cols, with nnz nonzero elements stored in COOrdinate matrix format
typedef struct matrix_coo_t {
    int nnz; // Number of nonzero elements
    int nrow; // Number of rows
    int ncol; // Number of columns
    int* row; // Array of row indicies
    int* col; // Array of column indicies
    double* val; // Array of the values
} matrix_coo_t;

// Function to create and initialize a random COO matrix with nnz nonzero values 
// uniformly distributed between min and max. Returns NULL on failure.
matrix_coo_t* matrix_coo_create_random_uniform(int nrow, int ncol, int nnz, double min, double max);

// Perform a deepcopy of a COO matrix and returns a pointer to the copy.
// Returns NULL on failure.
matrix_coo_t* matrix_coo_copy(const matrix_coo_t* m);

// Print a coo matrix.
void matrix_coo_print(const matrix_coo_t* m);

// Get the sparsity ratio of the matrix, a number between 0 and 1 computed as:
// number of zero elements over total number of elements.
double matrix_coo_get_sparsity(const matrix_coo_t* m);

// Free a dynamically allocated COO matrix and its contents.
void matrix_coo_free(matrix_coo_t* m);

// Reads a COO matrix from a .mtx file. 
// Returns a pointer to the dynamically allocated matrix, or NULL on failure.
// We assume that the .mtx file contains a real matrix in coordinate format. 
// Other formats are not supported.
matrix_coo_t* matrix_coo_read_from_mtx(const char* filename);

// Function that checks if the elements of a COO matrix are sorted following 
// the lexicographical order of the coordinates (row, col).
//
// Returns 1 if the condition is satisfied, otherwise returns 0.
int matrix_coo_is_sorted_by_coordinates(const matrix_coo_t* m);

// Function that sorts the elements of a COO matrix are sorted following 
// the lexicographical order of the coordinates (row, col).
//
// Internally it converts the matrix to an array of triplets, sorts it  
// with qsort and then converts it back.
//
// Returns 1 if the condition is satisfied, otherwise returns 0.
void matrix_coo_sort_by_coordinates(matrix_coo_t* m);

// A triplet of a COO matrix containing (row, col, val)
typedef struct coo_triplet_t {
    int row;
    int col;
    double val;
} coo_triplet_t;

// Returns a dynamically allocated vector of (row, col, val) triplets
// created starting from a matrix_coo. Returns NULL on failure.
coo_triplet_t* matrix_coo_to_triplets(matrix_coo_t* m);

// Compares two COO triplets in lexicographical coordinate order.
int coo_triplets_compare(const void* a, const void* b);

// Copies the COO triplets into the row, col, and val arrays of a COO matrix.
// The len of the triplets array is assumed to be equal to m->nnz.
void matrix_coo_copy_from_triplets(matrix_coo_t* m, coo_triplet_t* triplets);

#endif

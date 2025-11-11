#ifndef __CONVERSIONS_H__
#define __CONVERSIONS_H__

#include "matrix.h"
#include "matrix_coo.h"
#include "matrix_csr.h"
#include "matrix_csb.h"

// Create a dynamically allocated COO matrix from a dense matrix.
// Returns NULL on failure.
matrix_coo_t* matrix_coo_from_dense(const matrix_t* dense);

// Creates a dynamically allocated dense matrix from a COO matrix.
// Returns NULL on failure
matrix_t* matrix_dense_form_coo(const matrix_coo_t* coo);

// Creates a dynamically allocated CSR matrix from a COO matrix.
// If the COO matrix is not sorted in lexicographical coordinate order,
// we create a copy of it sort it before creating the CSR. 
// Returns NULL on failure.
matrix_csr_t* matrix_csr_from_coo(const matrix_coo_t* coo);

matrix_csb_t* matrix_csb_from_coo(const matrix_coo_t* coo);

#endif


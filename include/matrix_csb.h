#ifndef __MATRIX_CSB_H__
#define __MATRIX_CSB_H__

#include <stdint.h>
#include <math.h>

/*
    Matrix CSB struct implemented referencing the paper 
    "Parallel Sparse Matrix-Vector and Matrix-Transpose-Vector Multiplication 
    Using Compressed Sparse Blocks"  by A.Buluc et al.
*/
typedef struct matrix_csb_t {
    uint32_t nnz; 
    uint32_t nrow;
    uint32_t ncol;

    uint32_t block_size;  // This is β from the paper (each block is βxβ)
    uint32_t nblocks_row; // Calculated as ceil(nrow / (double)block_size)
    uint32_t nblocks_col; // Calculated as ceil(ncol / (double)block_size)

    /*
        Array of pointers to the start of each block's data.
        In other words, "blk_ptr[b] stores how many nnz elements have there been 
        in the blocks before block b".
        The total blocks number is nblocks_row * nblocks_col.
        Like CSR's row_ptr, this needs one extra element to know
        where the last block ends.
        Size: (nblocks_row * nblocks_col) + 1
    */
    uint32_t* blk_ptr; 

    // Array of nnz element values
    double* val; 

    /*
        Array of nnz packed (row, col) element positions.
        For a 32-bit integer:
        - High 16 bits: local row index (0 to β-1)
        - Low 16 bits:  local col index (0 to β-1)
        This implies our block_size (β) must be <= 65536. 
     */
    uint32_t* rowcol; 
} matrix_csb_t;

void matrix_csb_free(matrix_csb_t* m);

#endif

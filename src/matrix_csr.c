#include <stdio.h>
#include <stdlib.h>

#include "matrix_csr.h"

void matrix_csr_print(const matrix_csr_t *m)
{
    if (m == NULL || m->col == NULL || m->row_ptr == NULL || m->val == NULL)
    {
        fprintf(stderr, "Got NULL pointers in print function.\n");
        return;
    }

    printf("[\n");
    for (int i = 0; i < m->nrow; ++i)
    {
        for (int k = m->row_ptr[i]; k < m->row_ptr[i + 1]; ++k)
        {
            int j = m->col[k];
            double val = m->val[k];
            printf(" (%d,%d): %.2f\n", i, j, val);

            #ifdef MAX_PRINT_NUM
            // Optionally don't print all the nnz for big inputs
            if (k == MAX_PRINT_NUM - 1 && m->nnz > MAX_PRINT_NUM)
            {
                printf(" ...\n");
                printf("]\n");
                return;
            }
            #endif
        }
    }
    printf("]\n");
}

double matrix_csr_get_sparsity(const matrix_csr_t *m)
{
    if (m == NULL || m->row_ptr == NULL || m->col == NULL || m->val == NULL)
    {
        fprintf(stderr, "Got NULL pointers in get function.\n");
        exit(1);
    }

    return 1.0 - ((double)m->nnz) / (m->nrow * m->ncol);
}

int matrix_csr_get_num_empty_rows(const matrix_csr_t *m)
{
    if (m == NULL || m->row_ptr == NULL || m->col == NULL || m->val == NULL)
    {
        fprintf(stderr, "Got NULL pointers in get function.\n");
        exit(1);
    }

    int count = 0;
    for (int i = 0; i < m->nrow; ++i)
    {
        /* Each time we have two equal consecutive row_ptr elements, it means
           that the row has no nonzero elements, because row_ptr is the prefix
           sum of the count of nonzero elements per row. */
        if (m->row_ptr[i] == m->row_ptr[i + 1])
            count++;
    }

    return count;
}

int matrix_csr_get_num_rows_with_one_nonzero(const matrix_csr_t *m)
{
    if (m == NULL || m->row_ptr == NULL || m->col == NULL || m->val == NULL)
    {
        fprintf(stderr, "Got NULL pointers in get function.\n");
        exit(1);
    }

    int count = 0;
    for (int i = 0; i < m->nrow; ++i)
    {
        /* Each time we have two consecutive row_ptr elements that differ by
           exactly one, it means that the row has exactly one nonzero element,
           because row_ptr is the prefix sum of the count of nonzero elements
           per row. */
        if (m->row_ptr[i] + 1 == m->row_ptr[i + 1])
            count++;
    }

    return count;
}

double matrix_csr_get_average_nonzeros_per_row(const matrix_csr_t *m)
{
    if (m == NULL || m->row_ptr == NULL || m->col == NULL || m->val == NULL)
    {
        fprintf(stderr, "Got NULL pointers in get function.\n");
        exit(1);
    }
    return ((double)m->nnz / m->nrow);
}

#define HIST_PRINT_ELEMENTS_PER_ROW 15
void matrix_csr_print_nnz_per_row_histogram(const matrix_csr_t *m)
{
    if (m == NULL || m->row_ptr == NULL || m->col == NULL || m->val == NULL)
    {
        fprintf(stderr, "Got NULL pointers in print function.\n");
        return;
    }

    // Find the nnz of the row with the highest and lowest nnz
    int max_row_nnz = 0;
    int min_row_nnz = -1;
    for (int i = 0; i < m->nrow; ++i)
    {
        int row_nnz = m->row_ptr[i + 1] - m->row_ptr[i];
        max_row_nnz = (row_nnz > max_row_nnz) ? row_nnz : max_row_nnz;
        min_row_nnz = (row_nnz < min_row_nnz || min_row_nnz == -1) ? row_nnz : min_row_nnz;
    }
    min_row_nnz = (min_row_nnz == -1) ? 0 : min_row_nnz;

    // We have one slot for each possible number of nnz per row from min_row_nnz
    // to max_row_nnz inclusive
    int hist_size = max_row_nnz - min_row_nnz + 1;
    int *hist = calloc(hist_size + 1, sizeof(int));

    for (int i = 0; i < m->nrow; ++i)
    {
        int row_nnz = m->row_ptr[i + 1] - m->row_ptr[i];
        hist[row_nnz - min_row_nnz]++;
    }

    for (int ii = 0; ii < hist_size; ii += HIST_PRINT_ELEMENTS_PER_ROW)
    {
        printf("|%-8s", "Count");
        for (int i = ii; (i < hist_size) && (i < ii + HIST_PRINT_ELEMENTS_PER_ROW); ++i)
        {
            printf("|%4d", min_row_nnz + i);
        }
        printf("|\n");

        printf("|%-8s", "Row nnz");
        for (int i = ii; (i < hist_size) && (i < ii + HIST_PRINT_ELEMENTS_PER_ROW); ++i)
        {
            printf("|%4d", hist[i]);
        }
        printf("|\n\n");
    }

    free(hist);
}

void matrix_csr_free(matrix_csr_t *m)
{
    if (m == NULL)
        return;
    free(m->col);
    free(m->row_ptr);
    free(m->val);
    free(m);
}

double matrix_csr_get_spmv_bytes_moved(const matrix_csr_t *m)
{
    return (m->nrow * (2 * sizeof(int) + sizeof(double))) + (m->nnz * (2 * sizeof(double) + sizeof(int)));
}

double matrix_csr_get_spmv_total_flops(const matrix_csr_t *m)
{
    return 2.0 * m->nnz;
}
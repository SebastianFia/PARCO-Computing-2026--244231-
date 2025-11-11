#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"
#include "utils.h"

matrix_t *matrix_create_zeros(int nrow, int ncol)
{
    matrix_t *m = malloc(sizeof(matrix_t));
    if (m == NULL)
    {
        perror("Failed to allocate matrix");
        return NULL;
    }
    m->nrow = nrow;
    m->ncol = ncol;
    m->val = calloc(nrow * ncol, sizeof(double));
    if (m->val == NULL)
    {
        perror("Failed to allocate matrix");
        free(m);
        return NULL;
    }

    return m;
}

matrix_t *matrix_create_sparse(int nrow, int ncol, double p_nnz, double min, double max)
{
    matrix_t *m = matrix_create_zeros(nrow, ncol);
    if (m == NULL)
        return NULL;

    for (int i = 0; i < nrow; ++i)
    {
        for (int j = 0; j < ncol; ++j)
        {
            if (rand_double(0, 1) < p_nnz)
                m->val[i * ncol + j] = rand_double(min, max);
        }
    }

    return m;
}

void matrix_print(const matrix_t *m)
{
    if (m == NULL || m->val == NULL)
    {
        fprintf(stderr, "Got NULL pointers in print.\n");
        return;
    }

    printf("[\n");
    for (int i = 0; i < m->nrow; ++i)
    {
        printf(" [ ");
        for (int j = 0; j < m->ncol; ++j)
        {
            printf("%.2f, ", m->val[i * m->ncol + j]);

            #ifdef MAX_PRINT_NUM
            // Optionally don't print the full row for big inputs
            if (j == MAX_PRINT_NUM - 2 && m->ncol > MAX_PRINT_NUM + 1)
            {
                printf("..., ");
                j = m->ncol - 2; // Skip to last element
                continue;
            }
            #endif
        }
        printf("]\n");

        #ifdef MAX_PRINT_NUM
        // Optionally don't print all the rows for big inputs
        if (int i == MAX_PRINT_NUM - 2 && m->nrow > MAX_PRINT_NUM + 1)
        {
            printf(" ...\n");
            i = m->nrow - 2; // Skip to last element
            continue;
        }
        #endif
    }
    printf("]\n");
}

void matrix_free(matrix_t *m)
{
    if (m == NULL)
        return;
    free(m->val);
    free(m);
}

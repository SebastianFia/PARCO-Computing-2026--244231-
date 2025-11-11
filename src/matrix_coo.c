#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "matrix_coo.h"
#include "hashset.h"
#include "utils.h"

matrix_coo_t *matrix_coo_create_random_uniform(int nrow, int ncol, int nnz, double min, double max)
{
    if (nnz > nrow * ncol || nnz < 0 || nrow <= 0 || ncol <= 0)
    {
        fprintf(stderr, "Error: Invalid dimensions or nnz count.\n");
        return NULL;
    }

    matrix_coo_t *m = malloc(sizeof(matrix_coo_t));
    if (m == NULL)
    {
        perror("Failed to allocate matrix");
        return NULL;
    }

    m->nnz = nnz;
    m->nrow = nrow;
    m->ncol = ncol;
    m->row = malloc(nnz * sizeof(int));
    m->col = malloc(nnz * sizeof(int));
    m->val = malloc(nnz * sizeof(double));

    if (m->row == NULL || m->col == NULL || m->val == NULL)
    {
        perror("Failed to allocate matrix");
        free(m->row);
        free(m->col);
        free(m->val);
        free(m);
        return NULL;
    }

    // Hash set containing all the (flattened) coordinates we already sampled.
    // To reduce collisions with reasonable memory usage, it has a number of
    // buckets equal to the first prime greater than nnz.
    hashset_t *seen_flattened_coords = hashset_create(next_prime(nnz));
    if (seen_flattened_coords == NULL)
    {
        fprintf(stderr, "Failed to create set");
        free(m->row);
        free(m->col);
        free(m->val);
        free(m);
        return NULL;
    }

    // Generate nnz total values with distinct coordinates
    for (int k = 0; k < nnz; ++k)
    {
        // Sample coordinates. Resample if we have already seen them
        int i, j;
        do
        {
            i = randint(0, nrow - 1);
            j = randint(0, ncol - 1);
        } while (hashset_contains(seen_flattened_coords, i * ncol + j));

        hashset_insert(seen_flattened_coords, i * ncol + j);

        // Resample if the value is zero (whe need it to be nonzero)
        double val;
        do
        {
            val = rand_double(min, max);
        } while (val == 0.0);

        m->row[k] = i;
        m->col[k] = j;
        m->val[k] = val;
    }

    hashset_free(seen_flattened_coords);

    return m;
}

matrix_coo_t *matrix_coo_copy(const matrix_coo_t *m)
{
    if (m == NULL || m->val == NULL || m->row == NULL || m->col == NULL)
    {
        fprintf(stderr, "Got NULL pointers in copy function.\n");
        return NULL;
    }

    matrix_coo_t *new = malloc(sizeof(matrix_coo_t));

    if (new == NULL)
    {
        perror("Failed to allocate copy matrix");
        return NULL;
    }

    new->nrow = m->nrow;
    new->ncol = m->ncol;
    new->nnz = m->nnz;
    new->row = malloc(m->nnz * sizeof(int));
    new->col = malloc(m->nnz * sizeof(int));
    new->val = malloc(m->nnz * sizeof(double));

    if (new->row == NULL || new->col == NULL || new->val == NULL)
    {
        perror("Failed to allocate copy matrix");
        free(new->row);
        free(new->col);
        free(new->val);
        free(new);
        return NULL;
    }

    memcpy(new->row, m->row, m->nnz * sizeof(int));
    memcpy(new->col, m->col, m->nnz * sizeof(int));
    memcpy(new->val, m->val, m->nnz * sizeof(double));

    return new;
}

void matrix_coo_print(const matrix_coo_t *m)
{
    if (m == NULL || m->row == NULL || m->col == NULL || m->val == NULL)
    {
        fprintf(stderr, "Got NULL pointers in print function.\n");
        return;
    }

    printf("[\n");
    for (int k = 0; k < m->nnz; ++k)
    {
        printf(" (%d,%d): %.2f\n", m->row[k], m->col[k], m->val[k]);

        #ifdef MAX_PRINT_NUM
        // Optionally don't print all the elements for big inputs
        if (k == MAX_PRINT_NUM - 2 && m->nnz > MAX_PRINT_NUM)
        {
            printf(" ...\n");
            k = m->nnz - 2; // Skip to last element
            continue;
        }
        #endif
    }
    printf("]\n");
}

double matrix_coo_get_sparsity(const matrix_coo_t *m)
{
    if (m == NULL || m->row == NULL || m->col == NULL || m->val == NULL)
    {
        fprintf(stderr, "Got NULL pointers in get function.\n");
        exit(1);
    }
    return 1.0 - ((double)m->nnz) / (m->nrow * m->ncol);
}

void matrix_coo_free(matrix_coo_t *m)
{
    if (m == NULL)
        return;
    free(m->col);
    free(m->row);
    free(m->val);
    free(m);
}

#define READ_FROM_MTX_LINE_SIZE 1024
matrix_coo_t *matrix_coo_read_from_mtx(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "ERR: Unable to find file %s\n", filename);
        return NULL;
    }

    matrix_coo_t *m = NULL;
    int k = 0;
    int already_read_size = 0;
    char line[READ_FROM_MTX_LINE_SIZE];
    while (fgets(line, READ_FROM_MTX_LINE_SIZE, fp) != NULL)
    {
        // Skip commented lines and the header. We can ignore the infomation
        // contained in the header because we accept only one matrix format.
        if (line[0] == '%' || line[0] == '#')
            continue;

        if (!already_read_size)
        {
            // Try to read size (nrow, ncol and nnz)
            int nrow, ncol, nnz;
            if (sscanf(line, "%d %d %d", &nrow, &ncol, &nnz) != 3)
            {
                fprintf(stderr, "ERR: Failed to read matrix size from .mtx file\n");
                return NULL;
            };
            already_read_size = 1;

            // Create COO matrix
            m = malloc(sizeof(*m));
            m->nrow = nrow;
            m->ncol = ncol;
            m->nnz = nnz;
            m->row = malloc(sizeof(int) * nnz);
            m->col = malloc(sizeof(int) * nnz);
            m->val = malloc(sizeof(double) * nnz);
        }
        else
        {
            // Try to read each kth row, which will contain the data of the
            // kth nonzero element of the COO matrix.
            if (sscanf(line, "%d %d %lf", &m->row[k], &m->col[k], &m->val[k]) != 3)
            {
                fprintf(stderr, "ERR: Failed to read data line from .mtx file\n");
                return NULL;
            }

            // Convert from 1-based to 0-based indexing
            m->row[k]--;
            m->col[k]--;

            // Go to next line
            k++;

            // Don't read more data lines than the nnz specified in the file,
            // even if potentially there other lines to read.
            if (k == m->nnz)
                break;
        }
    };

    // Error if there weren't a line with the size
    if (!already_read_size)
    {
        fprintf(stderr, "ERR: Invalid .mtx file format.\n");
        return NULL;
    }

    if (k < m->nnz)
    {
        fprintf(stderr, "ERR: Less nnz elements than expected in .mtx file\n");
        return NULL;
    }

    return m;
}

int matrix_coo_is_sorted_by_coordinates(const matrix_coo_t *m)
{
    if (m == NULL || m->row == NULL || m->col == NULL || m->val == NULL)
    {
        fprintf(stderr, "Got NULL pointers in get-bool function.\n");
        exit(1);
    }

    int last_row = m->row[0];
    int last_col = m->col[0];
    for (int k = 1; k < m->nnz; ++k)
    {
        if ((m->row[k] < last_row) ||
            (m->row[k] == last_row && m->col[k] < last_col))
        {
            return 0;
        }

        last_row = m->row[k];
        last_col = m->col[k];
    }

    return 1;
}

void matrix_coo_sort_by_coordinates(matrix_coo_t *m)
{
    if (m == NULL || m->row == NULL || m->col == NULL || m->val == NULL)
    {
        fprintf(stderr, "Got NULLs in sorting function.\n");
        exit(1);
    }

    // Coo -> triplets
    coo_triplet_t *triplets = matrix_coo_to_triplets(m);
    if (triplets == NULL)
    {
        fprintf(stderr, "Failed to sort.\n");
        exit(1);
    }

    // Sort triplets
    qsort((void *)triplets, m->nnz, sizeof(coo_triplet_t), coo_triplets_compare);

    // Triplets -> coo
    matrix_coo_copy_from_triplets(m, triplets);

    free(triplets);
}

coo_triplet_t *matrix_coo_to_triplets(matrix_coo_t *m)
{
    if (m == NULL || m->row == NULL || m->col == NULL || m->val == NULL)
    {
        fprintf(stderr, "Got NULLs in conversion function\n");
        return NULL;
    }

    coo_triplet_t *triplets = malloc(sizeof(*triplets) * m->nnz);

    if (triplets == NULL)
    {
        fprintf(stderr, "Failed to allocate triplets\n");
        return NULL;
    }

    for (int k = 0; k < m->nnz; ++k)
        triplets[k] = (coo_triplet_t){m->row[k], m->col[k], m->val[k]};

    return triplets;
}

int coo_triplets_compare(const void *a, const void *b)
{
    coo_triplet_t *a_ = (coo_triplet_t *)a;
    coo_triplet_t *b_ = (coo_triplet_t *)b;

    // Ascending lexicographical order for the pair (row, col)
    return (a_->row == b_->row) ? (a_->col - b_->col) : (a_->row - b_->row);
}

void matrix_coo_copy_from_triplets(matrix_coo_t *m, coo_triplet_t *triplets)
{
    for (int k = 0; k < m->nnz; ++k)
    {
        m->row[k] = triplets[k].row;
        m->col[k] = triplets[k].col;
        m->val[k] = triplets[k].val;
    }
}

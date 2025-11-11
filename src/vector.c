#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "vector.h"
#include "utils.h"

vector_t *vector_create_zeros(int n)
{
    vector_t *v = malloc(sizeof(vector_t));
    if (posix_memalign((void **)&v->val, 64, sizeof(double) * n) != 0)
    {
        perror("Failed to allocate with posix_memalign");
        free(v);
        return NULL;
    }
    v->n = n;
    memset(v->val, 0, sizeof(double) * n);

    return v;
}

void vector_init_zeros(vector_t *v)
{
    if (v == NULL)
    {
        fprintf(stderr, "Got NULL pointers in init function.\n");
        return;
    }
    memset(v->val, 0, v->n * sizeof(double));
}

vector_t *vector_create_random_uniform(int n, double min, double max)
{
    vector_t *v = malloc(sizeof(vector_t));
    if (v == NULL)
    {
        perror("Failed to allocate vector");
        return NULL;
    }

    if (posix_memalign((void **)&v->val, 64, sizeof(double) * n) != 0)
    {
        perror("Failed to posix_memalign");
        free(v);
        return NULL;
    }
    v->n = n;

    for (int i = 0; i < n; ++i)
        v->val[i] = rand_double(min, max);

    return v;
}

void vector_free(vector_t *v)
{
    if (v == NULL)
        return;
    free(v->val);
    free(v);
}

void vector_print(const vector_t *v)
{
    if (v == NULL || v->val == NULL)
    {
        fprintf(stderr, "Got NULL pointers in print function.\n");
        return;
    }

    printf("[ ");
    for (int i = 0; i < v->n; ++i)
    {
        printf("%.2f, ", v->val[i]);

        #ifdef MAX_PRINT_NUM
        // Optionally don't print the full row for big inputs
        if (i == MAX_PRINT_NUM - 2 && v->n > MAX_PRINT_NUM + 1)
        {
            printf("..., ");
            i = v->n - 2; // Skip to last element
            continue;
        }
        #endif
    }
    printf("]\n");
}

double vector_mean_absolute_error(const vector_t *v1, const vector_t *v2)
{
    if (v1 == NULL || v2 == NULL || v1->val == NULL || v2->val == NULL)
    {
        fprintf(stderr, "Got NULL pointers in get function.\n");
        exit(EXIT_FAILURE);
    }

    if (v1->n != v2->n)
    {
        fprintf(stderr, "ERR: The two vectors must have the same size.\n");
        exit(EXIT_FAILURE);
    }

    return mean_absolute_error(v1->val, v2->val, v1->n);
}
#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <stdlib.h> 

// A vector struct containing a dynamically allocated array of doubles and the allocated size.
typedef struct vector_t {
    int n;
    double* val;
} vector_t;

// Creates a dynamically allocated vector of size n filled with zeros.
// Returns a pointer to it, or NULL on failure.
vector_t* vector_create_zeros(int n);

// Sets to zero all the values of a vector.
void vector_init_zeros(vector_t* v);

// Creates a dynamically allocated vector of size n.
// Each value is uniformly distributed between min and max.
// Returns a pointer to it, or NULL on failure.
vector_t* vector_create_random_uniform(int n, double min, double max);

// Frees a dynamically allocated vector and its elements.
// Does nothing on NULLs.
void vector_free(vector_t* v);

// Prints a vector.
void vector_print(const vector_t* v);

// Returns the mean absolute error between two vectors of the same length.
double vector_mean_absolute_error(const vector_t* v1, const vector_t* v2);

#endif

#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdlib.h>

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif 

// Generates a random int between min and max inclusive.
int randint(int min, int max);

// Generates a random double uniformly distributed between min and max.
double rand_double(double min, double max);

// Return the mean absolute error between two double arrays of same size
double mean_absolute_error(double* a, double* b, int n);

int is_prime(int n);

// Return the first prime >= than n.
int next_prime(int n);

// Checks if the env variable var_name exists AND it equals var_value
int env_equals(const char* var_name, const char* var_value);

// Safely retrieve an integer from an environment variable.
// Returns default_value on failure.
int get_int_from_env(const char *var_name, int default_value);

// Get current milliseconds.
double now_millis();

#endif

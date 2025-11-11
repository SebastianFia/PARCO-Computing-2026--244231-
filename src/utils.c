#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <errno.h>

#include "utils.h"

double rand_double(double min, double max)
{
    return min + (max - min) * (double)rand() / RAND_MAX;
}

int randint(int min, int max)
{
    return min + rand() % (max - min + 1);
}

double mean_absolute_error(double *a, double *b, int n)
{
    double mae = 0;
    int i;

    for (i = 0; i < n; ++i)
        mae += fabs(a[i] - b[i]);

    return mae / n;
}

int is_prime(int n)
{
    if (n <= 1)
        return 0;

    int end = (int)pow(n, 0.5);
    for (int i = 2; i <= end; ++i)
    {
        if (n % i == 0)
            return 0;
    }

    return 1;
}

int next_prime(int n)
{
    int k = (n % 2 == 0) ? (n + 1) : n; // First odd number >= n
    while (1)
    {
        if (is_prime(k))
            return k;
        k += 2; // Check next odd number
    }
}

int env_equals(const char *var_name, const char *var_value)
{
    char *val = getenv(var_name);
    return val && strcmp(val, var_value) == 0;
}

int get_int_from_env(const char *var_name, int default_value) {
    char *env_value = NULL;
    char *endptr = NULL;
    long result;

    env_value = getenv(var_name);

    if (env_value == NULL)
        return default_value;

    errno = 0; 
    result = strtol(env_value, &endptr, 10);
    
    if (errno == ERANGE) 
    {
        fprintf(stderr, "Error: %s value '%s' is out of range for long.\n", 
            var_name, env_value);
        return default_value;
    }
    
    if (endptr == env_value || *endptr != '\0') 
    {
        fprintf(stderr, "Error: %s value '%s' is not a valid integer.\n", 
            var_name, env_value);
        return default_value;
    }
    
    return (int)result;
}

double now_millis()
{
    return omp_get_wtime() * 1e3;
}
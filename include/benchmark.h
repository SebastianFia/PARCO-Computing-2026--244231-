#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

#include <omp.h>
#include <stdlib.h>
#include <time.h>

// Comparison function for sorting doubles in ascending order.
int compare_doubles(const void *a, const void *b);

//  Benchmark a function.
//  Returns the pth percentile of the measured times.
double benchmark_function(
    const char* name, // The benchmark name that is displayed when printing the results
    int iters, // Number of iterations for which to run the benchmark
    double p, // percentile of time to return (e.g. 0.5 is the median time, 0.9 is the time higher than 90%)
    void* (*function)(void*), // The function to benchmark
    void* args // The args for the function to benchmark
);

#endif

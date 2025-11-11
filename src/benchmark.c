#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "utils.h"
#include "benchmark.h"

int compare_doubles(const void *a, const void *b) {
    double a_ = *(const double *)a;
    double b_ = *(const double *)b;
    
    if (a_ < b_) return -1;
    else if (a_ == b_) return 0;
    else return 1;
}

double benchmark_function(
    const char* name, // The benchmark name that is displayed when printing the results
    int iters, // Number of iterations for which to run the benchmark
    double p, /* percentile time to return (e.g. 0.5 is the median time, 0.9 is the time higher than 90%) */
    void* (*function)(void*), // The function to benchmark
    void* args // The args for the function to benchmark
) {
    // warm-up
    function(args);
    double *times = malloc(sizeof(double) * iters);
    for (int t = 0; t < iters; ++t) {
        double t0 = now_millis();
        function(args);
        double t1 = now_millis();
        double dt = t1 - t0;
        times[t] = dt;
    }
    qsort(times, iters, sizeof(double), compare_doubles);

    double time_pth_percentile = times[(int)(iters * p)];

    if (env_equals("PRINT_VERBOSE", "true")) {
        printf(
            "%-18s: %8.4f ms, %dth percentile time over %d iteration(s)\n", 
            name, time_pth_percentile, (int)(100*p), iters
        );
    }

    free(times);

    return time_pth_percentile;
}
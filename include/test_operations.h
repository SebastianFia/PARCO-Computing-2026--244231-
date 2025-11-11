#ifndef __TEST_OPERATIONS_H__
#define __TEST_OPERATIONS_H__

#ifndef USE_DENSE
/*  If USE_DENSE is set to 1 we also allocate a full dense version of the sparse 
    matrix, benchmark the dense versions of the operations.

    We typically set USE_DENSE to 0 if we expect the dense matrix to use up too 
    much memory. */
#define USE_DENSE 0
#endif

#ifndef USE_DEFAULT_RANDOM_SEED
/* If 0 we generate a random seed based on the current time. */
#define USE_DEFAULT_RANDOM_SEED 0 
#endif

#ifndef DEFAULT_RANDOM_SEED
#define DEFAULT_RANDOM_SEED 42 /* Default random seed for reproducibility */
#endif

#ifndef READ_MATRIX
#define READ_MATRIX 1 /* If 1 we read the matrix from .mtx, else generate it */
#endif

#ifndef MIN_RAND_DOUBLE
#define MIN_RAND_DOUBLE -1.0
#endif

#ifndef MAX_RAND_DOUBLE
#define MAX_RAND_DOUBLE 1.0
#endif

#ifndef NROW
#define NROW (int)1e5 /* Num of rows if we generate the matrix */ 
#endif

#ifndef NCOL
#define NCOL (int)1e5 /* Num of cols if we generate the matrix */
#endif

#ifndef NNZ
#define NNZ (int)1e6 /* Num of nonzeros if we generate the matrix */
#endif 

#ifndef DEFAULT_BENCHMARK_PERCENTILE
#define DEFAULT_BENCHMARK_PERCENTILE 0.9
#endif

#ifndef DEFAULT_BENCHMARK_ITERATIONS
#define DEFAULT_BENCHMARK_ITERATIONS 10
#endif
    

/* Test function that runs all the operations and compares them. Returns 
EXIT_FAILURE on failure, else EXIT_SUCCESS. */
int test_operations();

#endif

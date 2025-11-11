#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <stdbool.h>

#include "test_operations.h"
#include "vector.h"
#include "matrix.h"
#include "matrix_coo.h"
#include "matrix_csr.h"
#include "matrix_csb.h"
#include "conversions.h"
#include "utils.h"
#include "operations.h"
#include "benchmark.h"

int test_operations()
{
    // Global exit status variable. Is set to EXIT_FAILURE only on error.
    int exit_status = EXIT_SUCCESS;

    //  ------------------------------------------------------------------------
    //  --------------------------READ ENV VARIABLES----------------------------
    //  ------------------------------------------------------------------------

    const bool print_verbose = env_equals("PRINT_VERBOSE", "true");

    const bool print_hist = env_equals("PRINT_NNZ_PER_ROW_HIST", "true");

    const char *benchmark_iterations_str = getenv("BENCHMARK_ITERATIONS");
    const int benchmark_iterations = benchmark_iterations_str ? 
        atoi(benchmark_iterations_str) : DEFAULT_BENCHMARK_ITERATIONS;

    const char *benchmark_percentile_str = getenv("BENCHMARK_PERCENTILE");
    const double benchmark_percentile = benchmark_percentile_str ?
        atof(benchmark_percentile_str) : DEFAULT_BENCHMARK_PERCENTILE;

    if (print_verbose)
    {
        printf("Running program with %s threads and schedule (%s).\n",
            getenv("OMP_NUM_THREADS"),
            getenv("OMP_SCHEDULE"));
    }

    //  ------------------------------------------------------------------------
    //  -------------------CREATE MATRICES AND VECTORS--------------------------
    //  ------------------------------------------------------------------------

    if (print_verbose)
        printf("\n---------------CREATE MATRICES AND VECTORS---------------\n");

#if USE_DEFAULT_RANDOM_SEED
    srand(DEFAULT_RANDOM_SEED); // For reproducibility
#else
    srand(time(NULL)); // To get different results each time
#endif

#if READ_MATRIX
    char *mtx_file_path = getenv("MTX_FILE_PATH");
    if (mtx_file_path == NULL)
    {
        fprintf(stderr, "No MTX_FILE_PATH env variable found");
        exit(EXIT_FAILURE);
    }
    if (print_verbose)
        printf("Reading random coo matrix from \"%s\"...\n", mtx_file_path);
    matrix_coo_t *coo = matrix_coo_read_from_mtx(mtx_file_path);
#else
    if (print_verbose)
        printf("Creating random coo matrix...\n");
    matrix_coo_t *coo = matrix_coo_create_random_uniform(
        NROW, NCOL, NNZ, MIN_RAND_DOUBLE, MAX_RAND_DOUBLE);
#endif /* READ_MATRIX */

    if (print_verbose)
        printf("Creating csr matrix...\n");
    matrix_csr_t *csr = matrix_csr_from_coo(coo);

    if (print_verbose)
        printf("Creating csb matrix...\n");
    matrix_csb_t *csb = matrix_csb_from_coo(coo);

    if (print_verbose)
        printf("Creating vectors...\n");
    vector_t *x = vector_create_random_uniform(
        coo->ncol, MIN_RAND_DOUBLE, MAX_RAND_DOUBLE);
    vector_t *y = vector_create_zeros(coo->ncol);
    vector_t *y_tmp = vector_create_zeros(coo->ncol);

    int failed_to_create = (coo == NULL || csr == NULL || csb == NULL 
        || y == NULL || y_tmp == NULL);

#if USE_DENSE
    if (print_verbose)
        printf("Creating dense matrix...\n");
    matrix_t *dense = matrix_dense_form_coo(coo);
    failed_to_create = failed_to_create || (dense == NULL);
#endif

    if (failed_to_create)
    {
        fprintf(stderr, "Failed to create all the matrices and vectors.\n");
        exit_status = EXIT_FAILURE;
        goto cleanup;
    }

    //  ------------------------------------------------------------------------
    //  -----------------------DISPLAY MATRIX PROPERTIES------------------------
    //  ------------------------------------------------------------------------
    if (print_verbose)
    {
        printf("\n-------------------MATRIX PROPERTIES---------------------\n");

        double sparsity = matrix_coo_get_sparsity(coo);
        double nonzero_density = 1. - sparsity;
        int total_elements = coo->nrow * coo->ncol;
        int nrows_empty = matrix_csr_get_num_empty_rows(csr);
        int nrows_one_nonzero = matrix_csr_get_num_rows_with_one_nonzero(csr);
        int nrows_many_nonzeros = csr->nrow - nrows_empty - nrows_one_nonzero;
        double avg_nnz_per_row = matrix_csr_get_average_nonzeros_per_row(csr);

        printf("Sparsity: %.8lf\n", sparsity);
        printf("Approx nonzero density: %1.1e\n", nonzero_density);
        printf("Approx total elements (dense): %1.1e\n", (double)total_elements);
        printf("Num rows: %d\n", csr->nrow);
        printf("Num nonzeros: %d\n", csr->nnz);
        printf("Num rows with no nonzeros: %d\n", nrows_empty);
        printf("Num rows with one nonzero: %d\n", nrows_one_nonzero);
        printf("Num rows with more than one nonzero: %d\n", nrows_many_nonzeros);
        printf("Approx average nonzeros per row: %1.1e\n", avg_nnz_per_row);

        if (print_hist)
        {
            printf("\nNnz per row hist:\n\n");
            matrix_csr_print_nnz_per_row_histogram(csr);
            printf("\n");
        }
    }

    //  ------------------------------------------------------------------------
    //  ------------------------BENCHMARK OPERATIONS----------------------------
    //  ------------------------------------------------------------------------
    if (print_verbose)
        printf("\n-------------------BENCHMARK-----------------------------\n");

    /*  Note that we will write multiple times to the y vector during the
        benchmarks, without ever resetting it to be filled with zeros. However
        this is not a problem, since we are benchmarking only the duration of
        the operations. */

#if USE_DENSE
    mat_vec_mul_args_t args_dense = (mat_vec_mul_args_t){dense, x, y};
    double duration_dense = benchmark_function(
        "Dense serial", benchmark_iterations, benchmark_percentile,
        mat_vec_mul_wrapper, (void *)&args_dense);
    double duration_dense_omp = benchmark_function(
        "Dense-Omp", benchmark_iterations, benchmark_percentile,
        mat_vec_mul_omp_wrapper, (void *)&args_dense);
#endif

    spmv_coo_args_t args_coo = (spmv_coo_args_t){coo, x, y};
    double duration_coo = benchmark_function(
        "COO serial", benchmark_iterations, benchmark_percentile,
        spmv_coo_wrapper, (void *)&args_coo);
    double duration_coo_omp = benchmark_function(
        "COO-Omp", benchmark_iterations, benchmark_percentile,
        spmv_coo_omp_wrapper, (void *)&args_coo);

    spmv_csr_args_t args_csr = (spmv_csr_args_t){csr, x, y};
    double duration_csr = benchmark_function(
        "CSR serial", benchmark_iterations, benchmark_percentile,
        spmv_csr_wrapper, (void *)&args_csr);
    double duration_csr_omp = benchmark_function(
        "CSR-Omp", benchmark_iterations, benchmark_percentile,
        spmv_csr_omp_wrapper, (void *)&args_csr);

    spmv_csb_args_t args_csb = (spmv_csb_args_t){csb, x, y};
    double duration_csb = benchmark_function(
        "CSB serial", benchmark_iterations, benchmark_percentile,
        spmv_csb_wrapper, (void *)&args_csb);
    double duration_csb_omp = benchmark_function(
        "CSB-Omp", benchmark_iterations, benchmark_percentile,
        spmv_csb_omp_wrapper, (void *)&args_csb);

#if USE_DENSE
    printf(
        "Speedup dense-serial -> dense-omp: %.2lfx\n",
        duration_dense / duration_dense_omp);
    printf(
        "Speedup dense-serial -> coo-serial: %.2lfx\n",
        duration_dense / duration_coo);
#endif
    printf(
        "Speedup coo-serial -> coo-omp: %.2lfx\n",
        duration_coo / duration_coo_omp);
    printf(
        "Speedup coo-serial -> csr-serial: %.2lfx\n",
        duration_coo / duration_csr);
    printf(
        "Speedup csr-serial -> csr-omp: %.2lfx\n",
        duration_csr / duration_csr_omp);
    printf(
        "Speedup csr-serial -> csb-serial: %.2lfx\n",
        duration_csr / duration_csb);
    printf(
        "Speedup csb-serial -> csb-omp: %.2lfx\n",
        duration_csb / duration_csb_omp);

    //  ------------------------------------------------------------------------
    //  -------------------------CORRECTNESS CHECK------------------------------
    //  ------------------------------------------------------------------------

    /*
        In this sections we check that the results of all of our variants of
        matrix vector multiplication give the same result as the serial COO
        case. When USE_DENSE is set to 1, it is actually more precise to say
        that we start by checking that the COO serial operation gives the same
        result as the simple serial dense operation, and then we check all other
        results against the COO serial one.
    */

    if (print_verbose)
    {
        printf("\n------------------CORRECTNESS CHECK----------------------\n");
        vector_init_zeros(y);
        spmv_coo(coo, x, y);
        double mae;

#if USE_DENSE
        vector_init_zeros(y_tmp);
        mat_vec_mul(dense, x, y_tmp);
        mae = vector_mean_absolute_error(y, y_tmp);
        printf("Mean absolute error between coo-serial and dense-serial y: %lf\n", mae);

        vector_init_zeros(y_tmp);
        mat_vec_mul_omp(dense, x, y_tmp);
        mae = vector_mean_absolute_error(y, y_tmp);
        printf("Mean absolute error between coo-serial and dense-omp y: %lf\n", mae);
#endif

        vector_init_zeros(y_tmp);
        spmv_coo_omp(coo, x, y_tmp);
        mae = vector_mean_absolute_error(y, y_tmp);
        printf("Mean absolute error between coo-serial and coo-omp y: %lf\n", mae);

        vector_init_zeros(y_tmp);
        spmv_csr(csr, x, y_tmp);
        mae = vector_mean_absolute_error(y, y_tmp);
        printf("Mean absolute error between coo-serial and csr-serial y: %lf\n", mae);

        vector_init_zeros(y_tmp);
        spmv_csr_omp(csr, x, y_tmp);
        mae = vector_mean_absolute_error(y, y_tmp);
        printf("Mean absolute error between coo-serial and csr-omp y: %lf\n", mae);

        vector_init_zeros(y_tmp);
        spmv_csb(csb, x, y_tmp);
        mae = vector_mean_absolute_error(y, y_tmp);
        printf("Mean absolute error between coo-serial and csb-serial y: %lf\n", mae);

        vector_init_zeros(y_tmp);
        spmv_csb_omp(csb, x, y_tmp);
        mae = vector_mean_absolute_error(y, y_tmp);
        printf("Mean absolute error between coo-serial and csb-omp y: %lf\n", mae);
    }

    //  ------------------------------------------------------------------------
    //  ---------------------------FREE RESOURCES-------------------------------
    //  ------------------------------------------------------------------------

// GOTO label for freeing resources and terminating the program
cleanup:;

    if (print_verbose)
        printf("\nFreeing resources...\n");
    matrix_coo_free(coo);
    matrix_csr_free(csr);
    matrix_csb_free(csb);
    vector_free(x);
    vector_free(y);
    vector_free(y_tmp);

#if USE_DENSE
    matrix_free(dense);
#endif

    if (print_verbose)
        printf("Program terminated.\n");

    return exit_status;
}
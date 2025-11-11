#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <utils.h>
#include <valgrind/valgrind.h>
#include <valgrind/callgrind.h>

#include "matrix_coo.h"
#include "matrix_csr.h"
#include "matrix_csb.h"
#include "vector.h"
#include "conversions.h"
#include "operations.h"
#include "test_operations.h"

#ifndef MIN_RAND_DOUBLE
#define MIN_RAND_DOUBLE -1.0
#endif

#ifndef MAX_RAND_DOUBLE
#define MAX_RAND_DOUBLE 1.0
#endif

int main() 
{
    // return test_operations();

    /* Global exit status variable. Is set to EXIT_FAILURE only on error. */
    int exit_status = EXIT_SUCCESS;


    // ---------------------------------------------------------------------------------------------
    // -----------------------------------READ ENV VARIABLES----------------------------------------
    // ---------------------------------------------------------------------------------------------
    const char* mtx_file_path = getenv("MTX_FILE_PATH");
    if (mtx_file_path == NULL) 
    {
        fprintf(stderr, "No MTX_FILE_PATH env variable found");
        exit(EXIT_FAILURE);
    }

    const int alg_is_csr = env_equals("ALG_NAME", "csr");
    const int alg_is_csb  = env_equals("ALG_NAME", "csb");
    if (!(alg_is_csr || alg_is_csb)) 
    {
        fprintf(stderr, "Please provide a supported ALG_NAME");
        exit(EXIT_FAILURE);
    }

    // ---------------------------------------------------------------------------------------------
    // -----------------------------CREATE MATRICES AND VECTORS-------------------------------------
    // ---------------------------------------------------------------------------------------------
    srand(time(NULL)); /* To have a different seed each time */

    matrix_coo_t* coo = matrix_coo_read_from_mtx(mtx_file_path);
    matrix_csr_t* csr = (alg_is_csr) ? matrix_csr_from_coo(coo) : NULL;
    matrix_csb_t* csb = (alg_is_csb) ? matrix_csb_from_coo(coo) : NULL;
    vector_t* x = vector_create_random_uniform(coo->ncol, MIN_RAND_DOUBLE, MAX_RAND_DOUBLE);
    vector_t* y = vector_create_zeros(coo->nrow);

    if (!coo || (alg_is_csb && !csr) || (alg_is_csb && !csb) || !x || !y)
    {
        fprintf(stderr, "Failed to create matrices and vectors.\n");
        exit_status = EXIT_FAILURE;
        goto cleanup;
    }
    

    // ---------------------------------------------------------------------------------------------
    // --------------------------------------BENCHMARK----------------------------------------------
    // ---------------------------------------------------------------------------------------------
    if (RUNNING_ON_VALGRIND) /*Measure cache miss rate*/
    {
        CALLGRIND_ZERO_STATS;

        if (alg_is_csr)
            spmv_csr_omp(csr, x, y);
        else if (alg_is_csb)
            spmv_csb_omp(csb, x, y);

        CALLGRIND_DUMP_STATS;
    }
    else  /*Measure time*/
    {
        /* Perform a warmup run */
        if (alg_is_csr)
            spmv_csr_omp(csr, x, y);
        else if (alg_is_csb)
            spmv_csb_omp(csb, x, y);

        double t0 = now_millis();
        if (alg_is_csr)
            spmv_csr_omp(csr, x, y);
        else if (alg_is_csb)
            spmv_csb_omp(csb, x, y);
        double t1 = now_millis();
        double duration_ms = t1 - t0;

        /* The duration in milliseconds is the only stdout output of the
        program, and it will be captured by the python script (if we are
        running with valgrind we don't output the duration) */
        fprintf(stdout, "%.4f\n", duration_ms);
    }



    // ---------------------------------------------------------------------------------------------
    // ---------------------------------------CLEANUP-----------------------------------------------
    // ---------------------------------------------------------------------------------------------
cleanup:; /* GOTO label for cleaning up resources and ending the program */

    matrix_coo_free(coo);
    matrix_csr_free(csr);
    matrix_csb_free(csb);
    vector_free(x);
    vector_free(y);

    return exit_status;
}
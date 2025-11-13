export OMP_PROC_BIND="true"
export OMP_PLACES="cores"
export OMP_SCHEDULE=${OMP_SCHEDULE:-"static, 64"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MTX_FILE_PATH=${MTX_FILE_PATH:-"matrices_data/twotone.mtx"}
export ALG_NAME=${ALG_NAME:-"csr"}
export TIMING_RUNS=${TIMING_RUNS:-10} # Number of times we will run the c program to get the 90th percentilie duration 
export VALGRIND_RUNS=${VALGRIND_RUNS:-1}  # Number of times we will run with callgrind to get the 90th percentile of D1 and LLd cache miss rates. 
                        # It defaults to 1 because callgrind is a deterministic tool and, despite the slight
                        # nondeterminism introduced by the use of multiple threads, we observed that the outputs have negligible differences.

export MATRICES_DIR="./matrices_data"
export OUTPUT_DIR="./experiments_output"

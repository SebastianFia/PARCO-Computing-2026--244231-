export MTX_FILE_PATH="matrices_data/1138_bus.mtx"
# export MTX_FILE_PATH="matrices_data/bcsstk17.mtx"
# export MTX_FILE_PATH="matrices_data/twotone.mtx"
# export MTX_FILE_PATH="matrices_data/nd12k.mtx"
# export MTX_FILE_PATH="matrices_data/cage14.mtx"

export OMP_NUM_THREADS=1

export OMP_SCHEDULE="static, 64"
# export OMP_SCHEDULE="dynamic,64"
# export OMP_SCHEDULE="guided"

export OMP_PROC_BIND="true"
export OMP_PLACES="cores"

export MATRICES_DIR="./matrices_data"
export OUTPUT_DIR="./experiments_output"
export ALG_NAME="csr"
export TIMING_RUNS=10 # Number of times we will run the c program to get the 90th percentilie duration 

export VALGRIND_RUNS=1  # Number of times we will run with callgrind to get the 90th percentile of D1 and LLd cache miss rates. 
                        # It defaults to 1 because callgrind is a deterministic tool and, despite the 
                        # nondeterminism introduced by the use of multiple threads, we observed that the outputs have negligible differences.

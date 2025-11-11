export MTX_FILE_PATH="matrices_data/1138_bus.mtx"
# export MTX_FILE_PATH="matrices_data/bcsstk17.mtx"
# export MTX_FILE_PATH="matrices_data/twotone.mtx"
# export MTX_FILE_PATH="matrices_data/nd12k.mtx"
# export MTX_FILE_PATH="matrices_data/cage14.mtx"

export OMP_NUM_THREADS=8

export OMP_SCHEDULE="static, 64"
# export OMP_SCHEDULE="dynamic,64"
# export OMP_SCHEDULE="guided"

export OMP_PROC_BIND="true"
export OMP_PLACES="cores"

export MATRICES_DIR="./matrices_data"
export OUTPUT_DIR="./experiments_output"
export ALG_NAME="csr"
export TIMING_RUNS=10 # Run each duration measurement this number of times to get 90th percentile
export VALGRIND_RUNS=1  # Run each cache miss rate measurement this number of times to get 90th percentile.
export OUTPUT_DIR="experiments_output"
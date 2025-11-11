#!/bin/bash
JOB_WALLTIME="05:30:00"
JOB_MEMORY="16gb"

for N_THREADS in 1 2 4 8 16 32; do
    for MTX_FILE_PATH in $MATRICES_DIR/*.mtx; do
        
        MATRIX_FILENAME=$(basename "$MTX_FILE_PATH")
        
        # Example: "twotone.mtx" -> "twotone"
        MATRIX_BASE_NAME=${MATRIX_FILENAME%.*}
        JOB_NAME="spmv_test_${N_THREADS}_${MATRIX_BASE_NAME}"

        echo "Submitting job: ${MATRIX_BASE_NAME} with $N_THREADS threads"

        qsub -N ${JOB_NAME} \
             -v JOB_N_THREADS=${N_THREADS},JOB_MTX_FILE_PATH=${MTX_FILE_PATH} \
             -l select=1:ncpus=${N_THREADS}:mem=${JOB_MEMORY},walltime=${JOB_WALLTIME} \
             scripts/job.pbs
    done
done
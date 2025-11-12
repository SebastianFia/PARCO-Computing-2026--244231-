#!/usr/bin/env bash
set -euo pipefail

# JOB_MTX_FILE_PATH and JOB_N_THREADS have to be set
: "${JOB_MTX_FILE_PATH:?Error: JOB_MTX_FILE_PATH is not set}"
: "${JOB_N_THREADS:?Error: JOB_N_THREADS is not set}"

JOB_WALLTIME=${JOB_WALLTIME:="05:30:00"}
JOB_MEMORY=${JOB_MEMORY:="16gb"}

# e.g. "matrices_data/twotone.mtx" -> "twotone.mtx"
JOB_MATRIX_FILENAME=$(basename "$JOB_MTX_FILE_PATH")

# e.g. "twotone.mtx" -> "twotone"
JOB_MATRIX_BASE_NAME=${JOB_MATRIX_FILENAME%.*}

JOB_NAME="spmv_${JOB_N_THREADS}_${JOB_MATRIX_BASE_NAME}"

echo "Submitting job: ${JOB_MATRIX_BASE_NAME} with $JOB_N_THREADS threads"
echo "  Walltime: ${JOB_WALLTIME}, Memory: ${JOB_MEMORY}"

qsub -N ${JOB_NAME} \
    -v JOB_N_THREADS=${JOB_N_THREADS},JOB_MTX_FILE_PATH=${JOB_MTX_FILE_PATH} \
    -l select=1:ncpus=${JOB_N_THREADS}:mem=${JOB_MEMORY},walltime=${JOB_WALLTIME} \
    -q short_cpuQ \
    scripts/job.pbs
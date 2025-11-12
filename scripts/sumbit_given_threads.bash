#!/usr/bin/env bash
set -euo pipefail

# JOB_N_THREADS has to be set
: "${JOB_N_THREADS:?Error: JOB_N_THREADS is not set}"

JOB_WALLTIME=${JOB_WALLTIME:="05:30:00"}
JOB_MEMORY=${JOB_MEMORY:="16gb"}
MATRICES_DIR=${MATRICES_DIR:="matrices_data"}

for JOB_MTX_FILE_PATH in "$MATRICES_DIR"/*.mtx; do
 JOB_N_THREADS=$JOB_N_THREADS \
 JOB_MTX_FILE_PATH=$JOB_MTX_FILE_PATH \
 JOB_WALLTIME=$JOB_WALLTIME \
 JOB_MEMORY=$JOB_MEMORY \
    bash scripts/submit_job.bash
done

#!/usr/bin/env bash
set -euo pipefail

JOB_WALLTIME=${JOB_WALLTIME:="05:30:00"}
JOB_MEMORY=${JOB_MEMORY:="16gb"}
MATRICES_DIR=${MATRICES_DIR:="matrices_data"}

for JOB_N_THREADS in 1 2 4 8 16 32; do
  for JOB_MTX_FILE_PATH in "$MATRICES_DIR"/*.mtx; do
    JOB_N_THREADS=$JOB_N_THREADS \
    JOB_MTX_FILE_PATH=$JOB_MTX_FILE_PATH \
    JOB_WALLTIME=$JOB_WALLTIME \
    JOB_MEMORY=$JOB_MEMORY \
      bash scripts/submit_job.bash
  done
done

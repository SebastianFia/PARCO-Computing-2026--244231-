#!/bin/bash
for N_THREADS in 1 2 4 8 16 32; do
  for MTX_FILE_PATH in $MATRICES_DIR/*.mtx; do
    qsub -v JOB_N_THREADS=$N_THREADS,JOB_MTX_FILE_PATH=$MTX_FILE_PATH job.pbs,
  done
done
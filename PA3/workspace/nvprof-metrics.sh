#!/usr/bin/bash
set -o errexit
set -o errtrace
METRICS="branch_efficiency,warp_execution_efficiency,warp_nonpred_execution_efficiency,gld_efficiency,gst_efficiency,shared_efficiency,"
(
  set -o xtrace
  srun --nodes=1 nvprof --metrics ${METRICS} ./benchmark 1000
)

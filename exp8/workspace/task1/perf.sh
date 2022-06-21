#!/usr/bin/bash
set -o errexit
set -o errtrace
for UNROLL_N in 1 2 4 8 16; do
  make clean
  make UNROLL_N=${UNROLL_N}
  (
    set -o xtrace
    srun ./main
  )
done

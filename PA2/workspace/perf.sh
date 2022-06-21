#!/usr/bin/bash

set -o errexit

export DAPL_DBG_TYPE=0

DATAPATH=stencil_data

for n in 1 2 4 8 16 28; do
  (
    set -o xtrace
    srun -N 1 --ntasks-per-node $n benchmark-mpi 7 512 512 512 100 ${DATAPATH}/stencil_data_512x512x512
  )
done

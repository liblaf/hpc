#!/usr/bin/bash
set -o errexit
set -o errtrace
SOURCE="test_smem.cu"
TARGET="test_smem"
echo -n >"perf_smem.md"
for BITWIDTH in 2 4 8; do
  for STRIDE in 1 2 4 8 16 32 64; do
    (
      set -o xtrace
      nvcc -DBITWIDTH=$BITWIDTH -DSTRIDE=$STRIDE $SOURCE -o $TARGET -O2 -code sm_60 -arch compute_60
      srun --exclusive ./$TARGET >>"perf_smem.md"
    )
  done
done

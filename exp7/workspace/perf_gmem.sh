#!/usr/bin/bash
set -o errexit
set -o errtrace
SOURCE="test_gmem.cu"
TARGET="test_gmem"
echo -n >"perf_gmem.md"
for STRIDE in 1 2 4 8; do
  (
    set -o xtrace
    nvcc -DSTRIDE=$STRIDE $SOURCE -o $TARGET -O2 -code sm_60 -arch compute_60
    srun --exclusive ./$TARGET >>"perf_gmem.md"
  )
done

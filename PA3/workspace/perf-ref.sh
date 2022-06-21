#!/usr/bin/bash
set -o errexit
set -o errtrace
TARGET="benchmark-ref"
make ${TARGET}
for n in 1000 2500 5000 7500 10000; do
  (
    set -o xtrace
    srun --nodes=1 "./${TARGET}" ${n}
  )
done

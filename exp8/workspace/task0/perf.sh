#!/usr/bin/bash
set -o errexit
set -o errtrace
for OPTION in 0 1 2 3 fast; do
  (
    set -o xtrace
    srun "./main_${OPTION}"
  )
done

#!/usr/bin/bash
set -o errexit
set -o errtrace
EVENTS="shared_ld_bank_conflict,shared_st_bank_conflict,"
(
  set -o xtrace
  srun --nodes=1 nvprof --events ${EVENTS} ./benchmark 1000
)

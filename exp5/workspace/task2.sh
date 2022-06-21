#!/usr/bin/bash task2.sh
set -x
srun -w conv3 -n 8 vtune -collect uarch-exploration -trace-mpi -result-dir uarch-exploration -- ./main 2020012872

#!/usr/bin/bash task1.sh
set -x
srun -n 8 vtune -collect hotspots -trace-mpi -result-dir hotspots -- ./main 2020012872

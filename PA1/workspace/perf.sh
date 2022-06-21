#!/bin/bash
set -o errexit
(
  set -x
  make
)
echo -n >res.txt
procs=("1 1" "1 2" "1 4" "1 8" "1 16" "2 32")
for number_count in 100000000; do
  for i in "${!procs[@]}"; do
    argv=(${procs[$i]})
    N=${argv[0]}
    n=${argv[1]}
    echo "+ srun -N $N -n $n ./odd_even_sort $number_count data/$number_count.dat" >>res.txt
    (
      set -x
      srun -N $N -n $n ./odd_even_sort $number_count data/$number_count.dat >>res.txt
    )
  done
done

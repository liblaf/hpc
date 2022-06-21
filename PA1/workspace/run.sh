#!/bin/bash

# 100: 1
# 1000: 3
# 10000: 17
# 100000: 18
# 1000000: 56

if [ $2 -lt 1000 ]; then
  n=1
elif [ $2 -lt 10000 ]; then
  n=3
elif [ $2 -lt 100000 ]; then
  n=17
elif [ $2 -lt 1000000 ]; then
  n=18
else
  n=56
fi

srun -n $n $*

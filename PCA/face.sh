#!/bin/bash

set -e

default_methods="BatchPCA RobustPCA LossGD LossMEG ConstPCA"

if [ "$#" -eq 0 ]; then
  methods_to_run=$default_methods
else
  methods_to_run="$@"
fi

for k in 2 3 5 10; do
    for method in $methods_to_run; do
        echo face $method, $k
        OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 python ./face.py $method --k $k --start 0 --end 30 --jobs 10 --overwrite --parallel
    done
done

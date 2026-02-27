#!/bin/bash

set -e

default_methods="BatchPCA RobustPCA LossGD LossMEG ConstPCA"

if [ "$#" -eq 0 ]; then
  methods_to_run=$default_methods
else
  methods_to_run="$@"
fi

for num in 50 100 300 500 1000; do
    for dim in 20 50 100; do
        for k in 2 3 5 10; do
            for method in $methods_to_run; do
                echo synthetic $method $num, $dim, $k
                OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python ./synthetic.py $method --dim $dim --num $num --k $k --start 0 --end 100 --jobs 100 --overwrite --parallel
            done
        done
    done
done

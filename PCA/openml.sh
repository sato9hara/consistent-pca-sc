#!/bin/bash

# micro-mass: https://www.openml.org/search?type=data&status=active&id=1514
# har: https://www.openml.org/search?type=data&status=active&id=1478
# gas-drift: https://www.openml.org/search?type=data&status=active&id=1476
# mnist_784: https://www.openml.org/search?type=data&sort=runs&id=554

set -e

default_methods="BatchPCA RobustPCA LossGD LossMEG ConstPCA"

if [ "$#" -eq 0 ]; then
  methods_to_run=$default_methods
else
  methods_to_run="$@"
fi

for k in 2 3 5 10; do
    for data in micro-mass har gas-drift mnist_784; do
        for method in $methods_to_run; do
            echo $data, $method, $k
            OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 python ./openml.py $method --data $data --k $k --classwise --start 0 --end 30 --jobs 15 --overwrite --parallel
        done
    done
done

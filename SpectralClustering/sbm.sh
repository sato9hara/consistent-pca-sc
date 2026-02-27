#!/bin/bash

for n in 50 100 500 1000; do
    for k in 2 5 10; do
        for method in BatchSC PCM PCQ; do
            echo $method $n, $k
            OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python sbm.py $method --k $k --n $n --start 0 --end 100 --jobs 100 --overwrite --parallel
        done
    done
done

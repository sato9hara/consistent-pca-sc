#!/bin/bash

for i in {0..5}; do
    for k in 2 3 5 10; do
        for method in BatchSC PCM PCQ; do
            echo $i, $method, $k
            OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python facebook.py $method --id $i --k $k --start 0 --end 100 --jobs 100 --overwrite --parallel
        done
    done
done
#!/bin/bash

for k in 2 3 5 10; do
    for method in BatchSC PCM PCQ; do
        echo $method, $k
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python email-eu.py $method --k $k --start 0 --end 100 --jobs 100 --overwrite --parallel
    done
done

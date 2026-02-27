#!/bin/bash

for i in {0..5}; do
    for k in 2 3 5 10; do
        for method in DSpSC; do
            echo $i, $method, $k
            OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python facebook.py $method --id $i --k $k --start 0 --end 20 --overwrite &
            OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python facebook.py $method --id $i --k $k --start 20 --end 40 --overwrite &
            OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python facebook.py $method --id $i --k $k --start 40 --end 60 --overwrite &
            OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python facebook.py $method --id $i --k $k --start 60 --end 80 --overwrite &
            OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python facebook.py $method --id $i --k $k --start 80 --end 100 --overwrite
        done
    done
done
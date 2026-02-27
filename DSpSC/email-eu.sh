#!/bin/bash

for k in 2 3 5 10; do
    for method in DSpSC; do
        echo $method, $k
        OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python email-eu.py $method --k $k --start 0 --end 20 --overwrite &
        OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python email-eu.py $method --k $k --start 20 --end 40 --overwrite &
        OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python email-eu.py $method --k $k --start 40 --end 60 --overwrite &
        OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python email-eu.py $method --k $k --start 60 --end 80 --overwrite &
        OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python email-eu.py $method --k $k --start 80 --end 100 --overwrite
    done
done

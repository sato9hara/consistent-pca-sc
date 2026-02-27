#!/bin/bash

for n in 50 100 500 1000; do
    for k in 2 5 10; do
        for method in DSpSC; do
            echo $method, $k
            OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python sbm.py $method --k $k --n $n --start 0 --end 20 --overwrite &
            OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python sbm.py $method --k $k --n $n --start 20 --end 40 --overwrite &
            OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python sbm.py $method --k $k --n $n --start 40 --end 60 --overwrite &
            OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python sbm.py $method --k $k --n $n --start 60 --end 80 --overwrite &
            OMP_NUM_THREADS=45 MKL_NUM_THREADS=45 python sbm.py $method --k $k --n $n --start 80 --end 100 --overwrite
        done
    done
done

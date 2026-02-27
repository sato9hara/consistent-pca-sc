import pathlib
import pickle
import joblib
import numpy as np
from scipy import sparse as sp
from copy import deepcopy

import sys
sys.path.append('../')
from config import get_SC

def gen_sbm(k, n, p, q, seed=0):
    np.random.seed(seed)
    labels = np.repeat(np.arange(k), n)
    probs = np.full((k, k), q)
    np.fill_diagonal(probs, p)
    P = np.kron(probs, np.ones((n, n)))
    Q = np.random.rand(k*n, k*n)
    A = np.tril(Q < P, -1).astype(int)
    A = A + A.T
    return sp.lil_array(A), labels

def save_sbm(dn, k, n, p, q, trial=100, seed=0):
    A, labels = gen_sbm(k, n, p, q, seed)
    np.random.seed(seed)
    B = [deepcopy(A)]
    for t in range(trial):
        while True:
            ii, jj = np.random.randint(0, nn), np.random.randint(0, nn)
            if ii == jj:
                continue
            r = np.random.rand()
            if labels[ii] == labels[jj]:
                pp = p
            else:
                pp = q
            if A[ii, jj] == 0 and r < pp:
                A[ii, jj] = 1
                A[jj, ii] = 1
                break
        B.append(deepcopy(A))
    with open(dn.joinpath('sbm%03d.pkl' % (seed,)), 'wb') as f:
        pickle.dump((B, labels, p, q), f)

if __name__ == "__main__":
    k = [2, 5, 10]
    n = [50, 100, 500, 1000]
    p = 0.5
    q = 0.1
    trial = 100
    seeds = range(100)

    SC_data = get_SC('data')
    dn = SC_data.joinpath('sbm')
    for kk in k:
        for nn in n:
            dnx = dn.joinpath('k%02d_n%04d' % (kk, nn))
            dnx.mkdir(parents=True, exist_ok=True)
            gen_fn = lambda seed: save_sbm(dnx, kk, nn//kk, p, q, trial=trial, seed=seed)
            joblib.Parallel(n_jobs=-1)(joblib.delayed(gen_fn)(seed) for seed in seeds)

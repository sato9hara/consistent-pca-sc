import numpy as np
from numpy import typing as npt
from sklearn.metrics import pairwise_distances_argmin_min

def procrustes(B, ref):
    U, _, V = np.linalg.svd(ref.T @ B)
    return B @ (V.T @ U.T)

def sample_from_p(p, seed):
    np.random.seed(seed)
    while True:
        j = np.random.choice(p.size, 10000)
        t = np.random.rand(j.size)
        k = np.where(t < p[j])[0]
        if k.size == 0:
            continue
        else:
            j = j[k[0]]
            break
    return j

def D1sampling(X: npt.ArrayLike, k: int, seed: int=0) -> (npt.ArrayLike, npt.ArrayLike, npt.ArrayLike):
    np.random.seed(seed)
    ids = [np.random.randint(X.shape[0])]
    centers = [X[ids[0]]]
    for i in range(k-1):
        D = pairwise_distances_argmin_min(X, centers)[1]
        p = D / D.sum()
        i = sample_from_p(p, seed+i)
        ids.append(i)
        centers.append(X[i])
    return np.array(ids), np.array(centers), pairwise_distances_argmin_min(X, centers)[0]

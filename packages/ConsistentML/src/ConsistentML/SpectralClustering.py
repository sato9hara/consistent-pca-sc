import numpy as np
from numpy import typing as npt
from scipy import sparse as sp
from copy import deepcopy

from .utils_SC import procrustes, D1sampling

def graph_laplacian(A: npt.ArrayLike) -> npt.ArrayLike:
    return sp.diags(np.sum(A, axis=1), 0) - A

def LUv(v, L, U, lam=1.0):
    Lv = L @ v
    UTv = U.T @ v
    return Lv + lam * v - lam * (U @ UTv)

def eigsh(L, k):
    while True:
        try:
            D, U = sp.linalg.eigsh(L, k=k, which='SM')
            break
        except:
            pass
    return D, U

class __OnlineSC:
    def __init__(self, n_components: int, seed: int=0) -> None:
        self.n_components = n_components
        self.seed = seed
    
    def reset(self, A: npt.ArrayLike) -> None:
        self.n = A.shape[0]
        L = graph_laplacian(A)
        D, U = eigsh(L, k=self.n_components)
        idx = np.argsort(D)
        U = U[:, idx]

        self.W = [U]
        ids, centers, labels = self.sampling(U)
        self.I = [ids]
        self.C = [centers]
        self.Y = [labels]

    def update(self, A: npt.ArrayLike) -> None:
        pass

    def sampling(self, U: npt.ArrayLike) -> (npt.ArrayLike, npt.ArrayLike, npt.ArrayLike):
        return D1sampling(U, self.n_components, seed=self.seed)
        
    def obj(self, A: npt.ArrayLike) -> npt.ArrayLike:
        L = graph_laplacian(A)
        return np.array([np.sum((L @ W) * W) for W in self.W])
    
    def inconsistency(self) -> npt.ArrayLike:
        d = []
        for i in range(len(self.W) - 1):
            Pi = procrustes(self.W[i+1], ref=self.W[i])
            inc_embedding = np.sum((Pi - self.W[i])**2)
            inc_centroid = np.sum((self.C[i+1] - self.C[i])**2)
            inc_clustering = np.mean((np.abs(self.Y[i+1] - self.Y[i]) > 0))
            d.append((inc_embedding, inc_centroid, inc_clustering))
        return np.array(d) 

class BatchSC(__OnlineSC):
    def __init__(self, n_components: int, seed: int=0) -> None:
        super().__init__(n_components, seed)

    def update(self, A: npt.ArrayLike) -> None:
        if not hasattr(self, 'n'):
            self.reset(A)
            return
        
        # solve SC
        assert A.shape[0] == self.n
        L = graph_laplacian(A)
        _, U = eigsh(L, k=self.n_components)
        self.W.append(U)
        ids, centers, labels = self.sampling(U)
        self.I.append(ids)
        self.C.append(centers)
        self.Y.append(labels)

class PCQ(BatchSC):
    def __init__(self, n_components: int, seed: int=0, alpha: float=0.9) -> None:
        super().__init__(n_components, seed)
        self.alpha = alpha
    
    def update(self, A: npt.ArrayLike) -> None:
        if not hasattr(self, 'n'):
            self.reset(A)
            self.A_prev = A
            return
        
        # solve SC
        assert A.shape[0] == self.n
        L = graph_laplacian(A)
        L_prev = graph_laplacian(self.A_prev)
        _, U = eigsh(self.alpha * L + (1 - self.alpha) * L_prev, k=self.n_components)
        self.W.append(U)
        ids, centers, labels = self.sampling(U)
        self.I.append(ids)
        self.C.append(centers)
        self.Y.append(labels)
        self.A_prev = deepcopy(A)

class ConsistentSC(BatchSC):
    def __init__(self, n_components: int, seed: int=0, lam: float=0.1) -> None:
        super().__init__(n_components, seed)
        self.lam = lam

    def update(self, A: npt.ArrayLike) -> None:
        if not hasattr(self, 'n'):
            self.reset(A)
            return
        
        # solve Eig
        assert A.shape[0] == self.n
        L = graph_laplacian(A)
        op = lambda v: LUv(v, L, self.W[-1], lam=self.lam)
        Lop = sp.linalg.LinearOperator((self.n, self.n), matvec=op)
        _, U = eigsh(Lop, k=self.n_components)
        self.W.append(U)
        ids, centers, labels = self.sampling(U)
        self.I.append(ids)
        self.C.append(centers)
        self.Y.append(labels)

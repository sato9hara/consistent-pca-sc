import numpy as np
from numpy import typing as npt

from .utils_PCA import eigh, procrustes, proj_PGD, proj_MEG, update_PGD, update_MEG

class __OnlinePCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
    
    def reset(self, x: npt.ArrayLike) -> None:
        self.dim = x.shape[1]
        C = x.T @ x / x.shape[0]
        D, U = eigh(C)
        idx = np.argsort(D)[::-1][:self.n_components]
        self.W = [U[:, idx]]

    def update(self, x: npt.ArrayLike) -> None:
        pass
        
    def transform(self, x: npt.ArrayLike, i: int=-1) -> npt.ArrayLike:
        assert x.shape[1] == self.dim
        return x @ self.W[i]
    
    def reconstruct(self, x: npt.ArrayLike, i: int=-1) -> npt.ArrayLike:
        assert x.shape[1] == self.dim
        return (x @ self.W[i]) @ self.W[i].T
    
    def obj(self, x: npt.ArrayLike) -> npt.ArrayLike:
        assert x.shape[1] == self.dim
        C = x.T @ x / x.shape[0]
        return np.array([np.sum((C @ W) * W) for W in self.W])
    
    def inconsistency(self) -> npt.ArrayLike:
        d = []
        for i in range(len(self.W) - 1):
            Pi = procrustes(self.W[i+1], ref=self.W[i])
            d.append(np.sum((Pi - self.W[i])**2))
        return d
    
    def reconstruction_error(self, x: npt.ArrayLike, i: int=-1) -> npt.ArrayLike:
        assert x.shape[1] == self.dim
        return np.mean(np.sum((x - self.reconstruct(x, i))**2, axis=1))

class BatchPCA(__OnlinePCA):
    def __init__(self, n_components: int, normalize=False) -> None:
        super().__init__(n_components)
        self.normalize = normalize
    
    def update(self, x: npt.ArrayLike) -> None:
        if not hasattr(self, 'dim'):
            if self.normalize:
                x = x / np.linalg.norm(x, axis=1, keepdims=True)
            self.reset(x)
            return
        
        # solve PCA
        assert x.shape[1] == self.dim
        if self.normalize:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
        C = x.T @ x / x.shape[0]
        D, U = eigh(C)
        idx = np.argsort(D)[::-1][:self.n_components]
        self.W.append(U[:, idx])

class OnlineGD(__OnlinePCA):
    def __init__(self, n_components: int, update_fn, proj_fn, eta: float=1e-3) -> None:
        super().__init__(n_components)
        self.update_fn = update_fn
        self.proj_fn = proj_fn
        self.eta = eta
    
    def update(self, x: npt.ArrayLike) -> None:
        if not hasattr(self, 'dim'):
            self.reset(x)
            self.S = np.eye(self.dim) - self.eta * x.T @ x / x.shape[0]
            return
        
        # gradient descent
        assert x.shape[1] == self.dim
        C = x.T @ x / x.shape[0]
        D, U = self.update_fn(self.S, C, self.eta)

        # projection & pos-process
        D = self.proj_fn(D, self.dim - self.n_components)
        idx = np.argsort(D)[:self.n_components]
        self.W.append(U[:, idx])
        self.S = (U * D[np.newaxis, :]) @ U.T
        self.S = 0.5 * (self.S + self.S.T)


class LossGD(OnlineGD):
    def __init__(self, n_components: int, eta: float=1e-3) -> None:
        update_fn = lambda S, C, e: update_PGD(S, C, e)
        proj_fn = lambda u, e: proj_PGD(u, e)
        super().__init__(n_components, update_fn, proj_fn, eta=eta)


class LossMEG(OnlineGD):
    def __init__(self, n_components: int, eta: float=1e-3, max_iter=1000, tol=1e-8) -> None:
        update_fn = lambda S, C, e: update_MEG(S, C, e, eps=1e-8)
        proj_fn = lambda u, e: proj_MEG(u, e, max_iter=max_iter, tol=tol)
        super().__init__(n_components, update_fn, proj_fn, eta=eta)


class ConsistentPCA(__OnlinePCA):
    def __init__(self, n_components: int, lam: float=0.1) -> None:
        super().__init__(n_components)
        self.lam = lam

    def update(self, x: npt.ArrayLike) -> None:
        if not hasattr(self, 'dim'):
            self.reset(x)
            return
        
        # solve Eig
        assert x.shape[1] == self.dim
        A = x.T @ x / x.shape[0] + 0.5 * (self.lam / x.shape[0]) * self.W[-1] @ self.W[-1].T
        D, U = eigh(A)
        idx = np.argsort(D)[::-1][:self.n_components]
        self.W.append(U[:, idx])
    
class RobustPCA(__OnlinePCA):
    def __init__(self, n_components: int, lam: float=0.1, rho: float=0.1, max_iter: int=100, tol: float=1e-3) -> None:
        super().__init__(n_components)
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.rho = rho
    
    def soft_thresholding(self, X, threshold):
        return np.sign(X) * np.maximum(np.abs(X) - threshold, 0)
    
    def singular_value_thresholding(self, X, threshold):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        S = self.soft_thresholding(S, threshold)
        return U @ np.diag(S) @ Vt

    def update(self, x: npt.ArrayLike) -> None:
        if not hasattr(self, 'dim'):
            self.reset(x)
            return
        
        # ADMM
        assert x.shape[1] == self.dim
        L = np.zeros((x.shape[0], self.dim))
        S = np.zeros((x.shape[0], self.dim))
        Y = np.zeros((x.shape[0], self.dim))
        for _ in range(self.max_iter):
            L_new = self.singular_value_thresholding(x - S + Y / self.rho, 1 / self.rho)
            S_new = self.soft_thresholding(x - L_new + Y / self.rho, self.lam / self.rho)
            Y_new = Y + self.rho * (x - L_new - S_new)
            residual = np.linalg.norm(x - L_new - S_new, 'fro')
            if residual < self.tol:
                L = L_new
                S = S_new
                break
            L = L_new
            S = S_new
            Y = Y_new
        D, U = eigh(L)
        idx = np.argsort(D)[::-1][:self.n_components]
        self.W.append(U[:, idx])

import numba
import numpy as np

def symmetrize(A):
    return 0.5 * (A + A.T)

def eigh(A):
    D, U = np.linalg.eigh(symmetrize(A))
    return D, U

def procrustes(B, ref):
    U, _, V = np.linalg.svd(ref.T @ B)
    return B @ (V.T @ U.T)

# LossPGD
@numba.njit("float64[:](float64[:], float64)")
def quadratic_knapsack(u, eta):
    if np.clip(u, 0, 1).sum() <= eta:
        return np.clip(u, 0, 1)
    n = u.size
    m = np.zeros(n + np.sum(u >= 1) + 1)
    b = -np.inf
    j = 0
    for i in range(n):
        m[j] = u[i]
        j = j + 1
        if u[i] >= 1:
            m[j] = u[i] - 1
            j = j + 1
        else:
            b = max(b, u[i] - 1)
    m[-1] = b
    m = np.sort(m)[::-1]
    f = np.maximum(0, np.minimum(1, u[:, np.newaxis] - m[np.newaxis, :])).sum(axis=0)
    i = np.searchsorted(f, eta) - 1
    Jn, Kn, uK = 0, 0, 0.0
    for j in range(n):
        if u[j] - m[i + 1] <= 0:
            continue
        elif u[j] - m[i] >= 1:
            Jn = Jn + 1
        else:
            Kn = Kn + 1
            uK = uK + u[j]
    mu = (Jn + uK - eta) / Kn
    return np.maximum(0, np.minimum(1, u - mu))

@numba.njit("float64[:](float64[:], float64)")
def meg_linear_subproblem(grad, k):
    s = np.zeros_like(grad)
    idx = np.argsort(grad)
    total_sum = 0
    for i in idx:
        if total_sum + 1 <= k:
            s[i] = 1
            total_sum += 1
        else:
            s[i] = max(0, k - total_sum)
            break
    return s

@numba.njit("float64[:](float64[:], float64, int64, float64)")
def meg_frank_wolfe_numba(u, k, max_iter, tol):
    n = len(u)
    x = np.ones(n) * (k / n)
    for t in range(1, max_iter+1):
        grad = np.log(x) - u
        s = meg_linear_subproblem(grad, k)
        gamma = 2 / (t + 2)
        x_new = (1 - gamma) * x + gamma * s
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x

def meg_frank_wolfe(u, k, max_iter=100, tol=1e-6):
    return meg_frank_wolfe_numba(u, k, max_iter, tol)

def proj_PGD(D, k):
    return quadratic_knapsack(D.astype(float), k)

def proj_MEG(D, k, max_iter=1000, tol=1e-8):
    return meg_frank_wolfe(np.log(np.maximum(D.astype(float), 1e-8)), float(k), max_iter=max_iter, tol=tol)

def update_PGD(S, C, eta=1e-3):
    D, U = eigh(S - eta * C)
    return D, U

def update_MEG(S, C, eta=1e-3, eps=1e-8):
    D, U = eigh(S)
    logS = U @ (np.log(np.maximum(eps, D))[:, np.newaxis] * U.T)
    D, U = eigh(logS - eta * C)
    return np.exp(D), U

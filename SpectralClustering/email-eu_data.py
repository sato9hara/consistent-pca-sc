import argparse
import pathlib
import numpy as np
from scipy import sparse as sp
import numba

import sys
sys.setrecursionlimit(2000)

sys.path.append('../')
from config import get_SC

@numba.njit
def X2IJ(X, s):
    X = np.array_split(X, s)
    m = max([x.shape[0] for x in X])
    I = np.full((s, 2*m), -1, dtype=np.int64)
    J = np.full((s, 2*m), -1, dtype=np.int64)
    for t, x in enumerate(X):
        c = 0
        for i, j, _ in x:
            I[t, c] = i
            J[t, c] = j
            I[t, c+1] = j
            J[t, c+1] = i
            c = c + 2
    return I, J

def dfs(adj_matrix, start_node, visited, component):
    visited[start_node] = True
    component.append(start_node)
    for neighbor in range(adj_matrix.shape[0]):
        if adj_matrix[start_node, neighbor] == 1 and not visited[neighbor]:
            dfs(adj_matrix, neighbor, visited, component)

def connected_components_dfs(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    visited = [False] * num_nodes
    components = []
    for node in range(num_nodes):
        if not visited[node]:
            component = []
            dfs(adj_matrix, node, visited, component)
            components.append(component)
    return components

def get_email(s=100):
    with open('./email-Eu-core-temporal.txt', 'r') as f:
        x = f.read()
    X = []
    for t in x.split('\n'):
        try:
            a, b, c = t.split()
            X.append((int(a), int(b), int(c)))
        except:
            pass
    X = np.array(X)
    _, Y = np.unique(X[:, :2], return_inverse=True)
    X[:, :2] = Y.reshape(X[:, :2].shape)
    I, J = X2IJ(X, s)
    A = []
    for t in range(1, s+1):
        i = I[:t].flatten()
        i = i[i >= 0]
        j = J[:t].flatten()
        j = j[j >= 0]
        a = sp.csr_array((np.ones(i.size, dtype=np.int64), (i, j)), shape=(I.max()+1, I.max()+1)).sign().tolil()
        A.append(a)
    c = connected_components_dfs(A[0])
    c = np.sort(c[0])
    return [a[c, :][:, c] for a in A]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split', type=int)
    args = parser.parse_args()

    SC_data = get_SC('data')
    dn = SC_data.joinpath('email-eu')
    if not dn.exists():
        dn.mkdir(parents=True, exist_ok=True)
        A = get_email(args.split)
        for i, a in enumerate(A):
            fn = dn.joinpath('A%03d.npz' % (i,))
            sp.save_npz(fn, a.tocsr())
    
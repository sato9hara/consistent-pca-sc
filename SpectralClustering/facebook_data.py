import argparse
import numpy as np
from scipy import sparse as sp
import pickle
import joblib
import pathlib

import sys
sys.path.append('../')
from config import get_SC

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

def load_facebook(data_id, split=10, seed=0):
    data = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
    assert data_id < len(data)

    # load edges
    with open('./facebook/%d.edges' % data[data_id], 'r') as f:
        lines = f.readlines()
    edges = []
    for line in lines:
        edges.append([int(a) for a in line.strip().split(' ')])
    edges = np.array(edges)
    nodes = np.max(edges) + 1

    # shuffle and split edges
    np.random.seed(seed)
    np.random.shuffle(edges)
    idx = np.floor(split * np.linspace(0, 1, len(edges)+1)[:-1])
    A = []
    for i in range(split):
        edges_i = edges[idx <= i]
        row = edges_i[:, 0]
        col = edges_i[:, 1]
        values = np.ones(len(row))
        A.append(sp.coo_array((values, (row, col)), shape=(nodes, nodes)).tolil())
        A[-1] = A[-1].maximum(A[-1].T)
    c = connected_components_dfs(A[0])
    i = np.argmax([len(cc) for cc in c])
    c = np.sort(c[i])
    return [a[c, :][:, c] for a in A]

def save_facebook(dn, data_id, split=10, seed=0):
    dn = dn.joinpath('data%02d' % data_id)
    fn = dn.joinpath('%02d.pkl' % seed)
    A = load_facebook(data_id, split, seed)
    with open(fn, 'wb') as f:
        pickle.dump(A, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('split', type=int)
    args = parser.parse_args()
    SC_data = get_SC('data')
    dn = SC_data.joinpath('facebook')
    if not dn.exists():
        dn.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            print(i)
            dnn = dn.joinpath('data%02d' % i)
            dnn.mkdir(parents=True, exist_ok=True)
            joblib.Parallel(n_jobs=-1)(joblib.delayed(save_facebook)(dn, i, args.split, seed) for seed in range(100))

import argparse
import pathlib
import joblib
import numpy as np
from sklearn.datasets import fetch_openml

import sys
sys.path.append('../')
from config import get_PCA

from utils import normalize, get_methods, get_argparse, test_all

def get_openml_minibatch(data_name, num=1000, batch=1, seed=0):
    X, _ = fetch_openml(data_name, version=1, return_X_y=True)
    X = X.values
    num = min(num, X.shape[0])
    np.random.seed(seed)
    idx = np.random.permutation(X.shape[0])[:num]
    X = X[idx]
    y = np.repeat(np.arange(np.ceil(num / batch).astype(int)), batch)[:num]
    return normalize(X), y

def get_openml_classwise(data_name, num=1000, seed=0):
    X, y = fetch_openml(data_name, version=1, return_X_y=True)
    X = X.values
    np.random.seed(seed)
    unique_labels = np.unique(y)
    shuffled_labels = np.random.permutation(unique_labels)
    label_mapping = {label: idx for idx, label in enumerate(shuffled_labels)}
    y = np.array([label_mapping[label] for label in y])
    if num < X.shape[0]:
        idx = np.random.permutation(X.shape[0])[:num]
        X, y = X[idx], y[idx]
    idx = np.argsort(y)
    return normalize(X[idx]), y[idx]

if __name__ == '__main__':
    methods = get_methods()
    parser = get_argparse(methods)
    parser.add_argument('--data', type=str)
    parser.add_argument('--num', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--classwise', action='store_true')
    args = parser.parse_args()

    # output dir
    PCA_result = get_PCA('result')
    dn = PCA_result.joinpath('openml')
    if args.classwise:
        dn = dn.joinpath('classwise')
        data_loader = lambda seed: get_openml_classwise(args.data, num=args.num, seed=seed)
    else:
        dn = dn.joinpath('minibatch')
        data_loader = lambda seed: get_openml_minibatch(args.data, num=args.num, batch=args.batch, seed=seed)
    dn = dn.joinpath(args.data)
    dn.mkdir(parents=True, exist_ok=True)

    # test
    test_fn = lambda seed: test_data(dn, data_loader, args.method, k=args.k, seed=seed, overwrite=args.overwrite)
    test_all(args, dn, data_loader)

import argparse
import pathlib
import joblib
import numpy as np

import sys
sys.path.append('../')
from config import get_PCA

from utils import get_methods, get_argparse, test_all

def get_synthetic(dim, num, seed=0):
    S = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            S[i, j] = min(i+1, j+1) / dim
    np.random.seed(seed)
    return np.random.multivariate_normal(np.zeros(dim), S, num), np.arange(num)

if __name__ == '__main__':
    methods = get_methods()
    parser = get_argparse(methods)
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--num', type=int, default=100)
    args = parser.parse_args()

    # output dir
    PCA_result = get_PCA('result')
    dn = PCA_result.joinpath('synthetic')
    dn = dn.joinpath('dim%03d_num%04d' % (args.dim, args.num))
    dn.mkdir(parents=True, exist_ok=True)

    # test
    data_loader = lambda seed: get_synthetic(dim=args.dim, num=args.num, seed=seed)
    test_all(args, dn, data_loader)

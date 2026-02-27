import argparse
import pathlib
import pickle
import joblib
import numpy as np

import sys
sys.path.append('../')
from config import get_SC

from utils import get_methods, get_argparse, test_all

def load_data(k, n, seed):
    SC_data = get_SC('data')
    dn = SC_data.joinpath('sbm')
    fn = dn.joinpath('k%02d_n%04d' % (k, n)).joinpath('sbm%03d.pkl' % (seed,))
    with open(fn, 'rb') as o:
        A, _, _, _ = pickle.load(o)
    return A

if __name__ == '__main__':
    methods = get_methods()
    parser = get_argparse(methods)
    parser.add_argument('--n', type=int, default=100)
    args = parser.parse_args()

    # output dir
    SC_result = get_SC('result')
    dn = SC_result.joinpath('sbm').joinpath('k%02d_n%04d' % (args.k, args.n))
    dn.mkdir(parents=True, exist_ok=True)

    # test
    data_loader = lambda seed: load_data(args.k, args.n, seed=seed)
    test_all(args, dn, data_loader)

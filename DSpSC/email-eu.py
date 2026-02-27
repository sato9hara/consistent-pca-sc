import argparse
import pathlib
import joblib
import numpy as np
from scipy import sparse as sp

import sys
sys.path.append('../')
from config import get_SC

from utils import get_methods, get_argparse, test_all

def load_data(trial=20, seed=0):
    SC_data = get_SC('data')
    dn = SC_data.joinpath('email-eu')
    A = []
    for i in range(trial):
        fn = dn.joinpath('A%03d.npz' % (i,))
        A.append(sp.load_npz(fn))
    return A

if __name__ == '__main__':
    methods = get_methods()
    parser = get_argparse(methods)
    args = parser.parse_args()

    # output dir
    SC_result = get_SC('result')
    dn = SC_result.joinpath('email-eu')
    dn.mkdir(parents=True, exist_ok=True)

    # test
    data_loader = lambda seed: load_data(seed=seed)
    test_all(args, dn, data_loader)

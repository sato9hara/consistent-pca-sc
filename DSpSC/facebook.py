import argparse
import pathlib
import pickle
import joblib
import numpy as np

import sys
sys.path.append('../')
from config import get_SC

from utils import get_methods, get_argparse, test_all

def load_data(data_id, seed=0):
    SC_data = get_SC('data')
    dn = SC_data.joinpath('facebook')
    dn = dn.joinpath('data%02d' % data_id)
    with open(dn.joinpath('%02d.pkl' % seed), 'rb') as f:
        A = pickle.load(f)
    return A

if __name__ == '__main__':
    methods = get_methods()
    parser = get_argparse(methods)
    parser.add_argument('--id', type=int, default=0, choices=list(range(6)))
    args = parser.parse_args()

    # output dir
    SC_result = get_SC('result')
    dn = SC_result.joinpath('facebook').joinpath('id%02d' % args.id)
    dn.mkdir(parents=True, exist_ok=True)

    # test
    data_loader = lambda seed: load_data(args.id, seed=seed)
    test_all(args, dn, data_loader)

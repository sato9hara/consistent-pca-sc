import argparse
import pathlib
import joblib
import numpy as np
from sklearn.datasets import fetch_lfw_people

import sys
sys.path.append('../')
from config import get_PCA

from utils import normalize, get_methods, get_argparse, test_all

def get_face(min_faces_per_person=20, seed=0):
    X, y = fetch_lfw_people(return_X_y=True, min_faces_per_person=min_faces_per_person)
    np.random.seed(seed)
    unique_labels = np.unique(y)
    shuffled_labels = np.random.permutation(unique_labels)
    label_mapping = dict(zip(unique_labels, shuffled_labels))
    y = np.array([label_mapping[label] for label in y])
    idx = np.argsort(y)
    return normalize(X[idx]), y[idx]

if __name__ == '__main__':
    methods = get_methods()
    parser = get_argparse(methods)
    parser.add_argument('--min', type=int, default=20)
    args = parser.parse_args()

    # output dir
    PCA_result = get_PCA('result')
    dn = PCA_result.joinpath('face')
    dn.mkdir(parents=True, exist_ok=True)

    # test
    data_loader = lambda seed: get_face(min_faces_per_person=args.min, seed=seed)
    test_all(args, dn, data_loader)

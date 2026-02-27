import argparse
import pathlib
import joblib
from functools import partial
import numpy as np

from ConsistentML import BatchPCA, LossGD, LossMEG, ConsistentPCA

methods = {
    'BatchPCA': [(BatchPCA, 'batch')],
    'RobustPCA': [(partial(BatchPCA, normalize=True), 'robust')],
    'LossGD': [(partial(LossGD, eta=eta), 'eta%.8f' % (eta,)) for eta in np.logspace(-5, 0, 11)],
    'LossMEG': [(partial(LossMEG, eta=eta, tol=1e-8, max_iter=1000), 'eta%.8f' % (eta,)) for eta in np.logspace(-5, -1, 9)],
    'ConstPCA': [(partial(ConsistentPCA, lam=lam), 'lam%09.1f' % (lam,)) for lam in np.append(0, np.logspace(0, 6, 13))],
}

def get_methods():
    return methods

def get_argparse(methods):
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, choices=list(methods.keys()))
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=10)
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    return parser

def normalize(X, eps=1e-8):
    return (X - np.mean(X, axis=0)[np.newaxis, :]) / (np.std(X, axis=0)[np.newaxis, :] + eps)

def test_method(X, y, method, k=1, seed=0):
    pca = method(n_components=k)
    for i in range(np.unique(y).size):
        try:
            pca.update(X[y <= i])
        except Exception as e:
            print(seed, i)
            print(e)
    obj = pca.obj(X)
    inc = pca.inconsistency()
    rec = pca.reconstruction_error(X)
    return np.array(obj), np.array(inc), np.array(rec)

def test_data(dn, data_loader, method_name, k=1, seed=0, overwrite=False):
    dn = dn.joinpath(method_name).joinpath('k%02d' % (k,))
    dn.mkdir(parents=True, exist_ok=True)
    X, y = data_loader(seed=seed)
    for m, f in methods[method_name]:
        if (not overwrite) and dn.joinpath('%s_%03d.npz' % (f, seed)).exists():
            continue
        obj, inc, rec = test_method(X, y, m, k=k, seed=seed)
        np.savez_compressed(dn.joinpath('%s_%03d.npz' % (f, seed)), obj=obj, inc=inc, rec=rec)

def test_all(args, dn, data_loader):
    test_fn = lambda seed: test_data(dn, data_loader, args.method, k=args.k, seed=seed, overwrite=args.overwrite)
    if args.parallel:
        joblib.Parallel(n_jobs=args.jobs)(joblib.delayed(test_fn)(seed) for seed in range(args.start, args.end))
    else:
        for seed in range(args.start, args.end):
            test_fn(seed)

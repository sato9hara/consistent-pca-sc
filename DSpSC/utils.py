import argparse
from functools import partial
import joblib
import numpy as np

from DSpSC import DSpSC

methods = {
    'DSpSC': [(partial(DSpSC, sampling_constant=sc), 'sc%02d' % (sc,)) for sc in np.linspace(5, 25, 11).astype(int)],
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

def test_method(A, method, k=2, s=0):
    sc = method(k, s)
    for a in A:
        sc.update(a)
    obj = sc.obj(A[-1])
    inc = sc.inconsistency()
    return np.array(obj), np.array(inc), np.array(sc.Y), np.array(sc.C), np.array(sc.I)

def test_data(dn, data_loader, method_name, k=2, seed=0, overwrite=False):
    dn = dn.joinpath(method_name).joinpath('k%02d' % (k,))
    dn.mkdir(parents=True, exist_ok=True)
    A = data_loader(seed=seed)
    for m, f in methods[method_name]:
        if (not overwrite) and dn.joinpath('%s_%03d.npz' % (f, seed)).exists():
            continue
        obj, inc, Y, C, I = test_method(A, m, k=k, s=seed)
        np.savez_compressed(dn.joinpath('%s_%03d.npz' % (f, seed)), obj=obj, inc=inc, Y=Y, C=C, I=I)

def test_all(args, dn, data_loader):
    test_fn = lambda seed: test_data(dn, data_loader, args.method, k=args.k, seed=seed, overwrite=args.overwrite)
    if args.parallel:
        joblib.Parallel(n_jobs=args.jobs)(joblib.delayed(test_fn)(seed) for seed in range(args.start, args.end))
    else:
        for seed in range(args.start, args.end):
            test_fn(seed)
            

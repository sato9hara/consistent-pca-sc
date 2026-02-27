import numpy as np
from numpy import typing as npt
from scipy import sparse as sp
from copy import deepcopy

from ConsistentML import BatchSC, graph_laplacian, eigsh

from graph_tool import Graph
import graph_tool as gt
from graph_tool.all import Graph

from .DySC.spectral_clustering import *
from .DySC.contractedgraph import *
from .DySC.sparsifier import *

def find_differing_indices(A, B):
    diff = sp.tril(A != B)
    differing_indices = np.array(diff.nonzero()).T
    return differing_indices

class DSpSC(BatchSC):
    def __init__(self, n_components: int, seed: int=0, sampling_constant: int=5) -> None:
        super().__init__(n_components, seed)
        self.sampling_constant = sampling_constant
    
    def reset(self, A):
        g = Graph({i:a for i, a in enumerate(A.tolil().rows)}, directed=False)
        self.sparsifier = DynamicGraphSparsifier(g, sampling_constant=self.sampling_constant)
        self.sparsifier.create_sparsifier()
        B = gt.spectral.adjacency(self.sparsifier.sparsified_graph)
        B = sp.csr_array(B)
        super().reset(B)
        self.A_prev = deepcopy(A)

    def update_sparsifier(self, A):
        edges_to_add = find_differing_indices(self.A_prev.tolil(), A.tolil())
        self.sparsifier.update_sparsifier(edges_to_add, verbose=False)
    
    def sparsify(self, A):
        B = gt.spectral.adjacency(self.sparsifier.sparsified_graph)
        return sp.csr_array(B)

    def update(self, A: npt.ArrayLike) -> None:
        if not hasattr(self, 'n'):
            self.reset(A)
            return
        
        # solve SC
        assert A.shape[0] == self.n
        self.update_sparsifier(A)
        B = self.sparsify(A)
        L = graph_laplacian(B)
        _, U = eigsh(L, k=self.n_components)
        self.W.append(U)
        ids, centers, labels = self.sampling(U)
        self.I.append(ids)
        self.C.append(centers)
        self.Y.append(labels)
        self.A_prev = deepcopy(A)

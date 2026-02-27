from .PCA import (
    BatchPCA,
    LossGD,
    LossMEG,
    ConsistentPCA,
)
from .SpectralClustering import (
    graph_laplacian, 
    eigsh,
    __OnlineSC,
    BatchSC,
    PCQ,
    ConsistentSC
)
from .utils_SC import (
    procrustes,
    D1sampling
)

__version__ = '0.0.1'
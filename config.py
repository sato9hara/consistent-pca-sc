import pathlib

root_dir = "/your/directory/"
root_path = pathlib.Path(root_dir)
PCA_data = root_path.joinpath('data').joinpath('PCA')
PCA_result = root_path.joinpath('result').joinpath('PCA')
PCA_figure = root_path.joinpath('figure').joinpath('PCA')
SC_data = root_path.joinpath('data').joinpath('SpectralClustering')
SC_result = root_path.joinpath('result').joinpath('SpectralClustering')
SC_figure = root_path.joinpath('figure').joinpath('SpectralClustering')

def get_root():
    return root_path

def get_PCA(path='data'):
    assert path in ['data', 'result', 'figure']
    if path == 'data':
        return PCA_data
    elif path == 'result':
        return PCA_result
    elif path == 'figure':
        return PCA_figure

def get_SC(path='data'):
    assert path in ['data', 'result', 'figure']
    if path == 'data':
        return SC_data
    elif path == 'result':
        return SC_result
    elif path == 'figure':
        return SC_figure

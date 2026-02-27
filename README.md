# ConsistentML

This is an implementation of the paper "Consistent PCA and Spectral Clustering" at 
AISTATS'26.

As the part of our implementation, we used [Dynamic-Spectral-Clustering-With-Provable-Approximation-Guarantee](https://github.com/SteinarLaenen/Dynamic-Spectral-Clustering-With-Provable-Approximation-Guarantee).

## To Reproduces The Results
### Install
```
cd ./packages/ConsistentML
pip install .
cd ../DSpSC
conda env create -f environment.yml
conda activate dynamic_SC
pip install .
conda deactivate
```

### Before Experiment
* Edit `root_dir` of `config.py`.

#### PCA
```
./PCA.sh
```

#### Spectral Clustering
```
./SpectralClustering.sh
```
```
conda activate env
./DSpSC.sh
conda deactivate
```

## Figures
* PCA Experiments
    * [synthetic data](https://htmlpreview.github.io/?https://github.com/sato9hara/consistent-pca-sc/blob/main/figures/PCA/synthetic/plots.html)
    * [face](https://htmlpreview.github.io/?https://github.com/sato9hara/consistent-pca-sc/blob/main/figures/PCA/face/plots.html)
    * [openml](https://htmlpreview.github.io/?https://github.com/sato9hara/consistent-pca-sc/blob/main/figures/PCA/openml/plots.html)
* Spectral Clustering Experiments
    * [stochastic block model](https://htmlpreview.github.io/?https://github.com/sato9hara/consistent-pca-sc/blob/main/figures/SpectralClustering/sbm/plots.html)
    * [email](https://htmlpreview.github.io/?https://github.com/sato9hara/consistent-pca-sc/blob/main/figures/SpectralClustering/email-eu/plots.html)
    * [facebook](https://htmlpreview.github.io/?https://github.com/sato9hara/consistent-pca-sc/blob/main/figures/SpectralClustering/facebook/plots.html)
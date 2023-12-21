import matplotlib.pylab as plt
import pickle
import utils
import numpy as np
import scipy.io as sio
import scipy.signal as sps
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from pingouin import ttest


data = pickle.load(open('./acoustic_indices.pkl', 'rb'))
tab = np.asarray(np.squeeze(data['tabAcoustic_indices'],axis=1))
n_samples = tab.shape[0]

print(tab.shape)

# deux indices
plt.scatter(tab[:,16],tab[:,21], c=range(tab.shape[0]))
plt.xlabel('ACI')
plt.ylabel('BI')
plt.show()

# pca
n_components = 3
pca = PCA(n_components=n_components,whiten=True)
tab_pca = pca.fit_transform(tab)
print(tab_pca.shape)
plt.scatter(tab_pca[:,0],tab_pca[:,1], c=range(tab_pca.shape[0]))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# t-sne
n_components = 2
tsne = TSNE(n_components=n_components, random_state=42)
tab_tsne = tsne.fit_transform(tab)
print(tab_tsne.shape)
plt.scatter(tab_tsne[:,0],tab_tsne[:,1], c=range(tab_tsne.shape[0]))
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.show()

# umap
n_components = 100
umap = umap.UMAP(n_neighbors=n_components, min_dist=.9)
tab_umap = umap.fit_transform(tab)
print(tab_umap.shape)
plt.scatter(tab_umap[:,0],tab_umap[:,1], c=range(tab_umap.shape[0]))
plt.xlabel('umap1')
plt.ylabel('umap2')
plt.show()


# # scikit maad indices
# SPECTRAL_FEATURES=['MEANf','VARf','SKEWf','KURTf','NBPEAKS','LEQf',
# 'ENRf','BGNf','SNRf','Hf', 'EAS','ECU','ECV','EPS','EPS_KURT','EPS_SKEW','ACI',
# 'NDSI','rBA','AnthroEnergy','BioEnergy','BI','ROU','ADI','AEI','LFC','MFC','HFC',
# 'ACTspFract','ACTspCount','ACTspMean', 'EVNspFract','EVNspMean','EVNspCount',
# 'TFSD','H_Havrda','H_Renyi','H_pairedShannon', 'H_gamma', 'H_GiniSimpson','RAOQ',
# 'AGI','ROItotal','ROIcover']

# TEMPORAL_FEATURES=['ZCR','MEANt', 'VARt', 'SKEWt', 'KURTt',
#                'LEQt','BGNt', 'SNRt','MED', 'Ht','ACTtFraction', 'ACTtCount',
#                'ACTtMean','EVNtFraction', 'EVNtMean', 'EVNtCount']
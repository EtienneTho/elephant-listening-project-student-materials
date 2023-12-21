import matplotlib.pylab as plt
import pickle
import utils
import numpy as np
import scipy.io as sio
import scipy.signal as sps
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE # pip3 install scikit-learn
import umap # pip3 install umap-learn
from pingouin import ttest # pip3 install pingouin


data = pickle.load(open('./acoustic_indices.pkl', 'rb'))
print(np.asarray(data['tabAcoustic_indices']).shape)
tab = np.asarray(np.squeeze(data['tabAcoustic_indices'],axis=1))
n_samples = tab.shape[0]

# affichage
plt.plot(tab[:,16])
plt.title('Acoustic Complexity')
plt.show()
plt.plot(tab[:,21])
plt.title('Bioacoustic Index')
plt.show()

# tableaux par tranche d'âge
times_1 = tab[:int(n_samples/4),16]
times_2 = tab[int(n_samples/4)+1:int(n_samples/4)*2,16]
times_3 = tab[int(n_samples/4)*2+1:int(n_samples/4)*3,16]
times_4 = tab[int(n_samples/4)*3+1:,16]

# moyenne et écart-type pour chaque plage horaire
print('M='+str(np.mean(times_1))+', SD='+str(np.std(times_1))+', N='+str(times_1.shape[0]))
print('M='+str(np.mean(times_2))+', SD='+str(np.std(times_2))+', N='+str(times_2.shape[0]))
print('M='+str(np.mean(times_3))+', SD='+str(np.std(times_3))+', N='+str(times_3.shape[0]))
print('M='+str(np.mean(times_4))+', SD='+str(np.std(times_4))+', N='+str(times_4.shape[0]))

# t-test
print('(1/6) vs. (7/12):')
print(ttest(times_1,times_2))

print()
print('(1/6) vs. (13/18):')
print(ttest(times_1,times_3))

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
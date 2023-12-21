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
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.svm import SVC


tabStrf = []

data = pickle.load(open('./acoustic_indices.pkl', 'rb'))
tab = np.asarray(np.squeeze(data['tabAcoustic_indices'],axis=1))
n_samples = tab.shape[0]

# tableaux par tranche d'âge
times_1 = tab[:int(n_samples/4),16]
times_2 = tab[int(n_samples/4)+1:int(n_samples/4)*2,16]
times_3 = tab[int(n_samples/4)*2+1:int(n_samples/4)*3,16]
times_4 = tab[int(n_samples/4)*3+1:,16]

labels = np.zeros((1, n_samples), dtype=int)

# Assign values to segments without overlapping indices
labels[0, :int(n_samples / 4)] = 0
labels[0, int(n_samples / 4):int(n_samples / 2)] = 1
labels[0, int(n_samples / 2):int(3 * n_samples / 4)] = 2
labels[0, int(3 * n_samples / 4):] = 3

# classification
tabBAcc = []
Ntimes = 3
cv = StratifiedKFold(5, shuffle=True)

for iRepeat in range(Ntimes):
	print('Repeat #',str(iRepeat))
	X = []
	y = []
	X = tab
	y = np.squeeze(labels,axis=0)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=iRepeat)
	n_components = 3
	pca = PCA(n_components=n_components,whiten=True)
	X_pca_train = pca.fit_transform(X_train)
	tuned_parameters = { 'gamma':np.logspace(-3,3,num=3),'C':np.logspace(-3,3,num=3)}
	clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, n_jobs=-1, cv=cv,  pre_dispatch=6,
	         scoring='balanced_accuracy', verbose=False)
	clf.fit(X_pca_train, y_train)
	y_test_pred = clf.predict(pca.transform(X_test))

	# print(balanced_accuracy_score(Y_test,Y_test_pred))
	tabBAcc.append(balanced_accuracy_score(y_test,y_test_pred))
	print(confusion_matrix(y_test,y_test_pred))

print('Balanced Accuracy: M='+str(np.mean(tabBAcc))+', SD='+str(np.std(tabBAcc)))


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
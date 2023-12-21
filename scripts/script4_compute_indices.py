import matplotlib.pylab as plt
import pickle
import utils
import numpy as np
import scipy.io as sio
import scipy.signal as sps
import os
import librosa
import soundfile as sf
from maad import sound, features
from maad.util import (date_parser, plot_correlation_map,
                       plot_features_map, plot_features, false_Color_Spectro)

tabStrf = []

dossier = "./output/nn07c_20220603_000000/"


tabAcoustic_indices = []

fs = 16000

# scikit maad indices
SPECTRAL_FEATURES=['MEANf','VARf','SKEWf','KURTf','NBPEAKS','LEQf',
'ENRf','BGNf','SNRf','Hf', 'EAS','ECU','ECV','EPS','EPS_KURT','EPS_SKEW','ACI',
'NDSI','rBA','AnthroEnergy','BioEnergy','BI','ROU','ADI','AEI','LFC','MFC','HFC',
'ACTspFract','ACTspCount','ACTspMean', 'EVNspFract','EVNspMean','EVNspCount',
'TFSD','H_Havrda','H_Renyi','H_pairedShannon', 'H_gamma', 'H_GiniSimpson','RAOQ',
'AGI','ROItotal','ROIcover']

TEMPORAL_FEATURES=['ZCR','MEANt', 'VARt', 'SKEWt', 'KURTt',
               'LEQt','BGNt', 'SNRt','MED', 'Ht','ACTtFraction', 'ACTtCount',
               'ACTtMean','EVNtFraction', 'EVNtMean', 'EVNtCount']

# Parcourir le dossier et ses sous-dossiers
for dossier_racine, sous_dossiers, fichiers in os.walk(dossier):
    print(sorted(fichiers))
    for fichier in sorted(fichiers):
    # Vérifier si le fichier a l'extension .wav
     if fichier.endswith(".wav"):
        # Chemin complet du fichier .wav
        print(fichier)
        chemin_fichier_wav = os.path.join(dossier_racine, fichier)
        print("Fichier .wav trouvé :", chemin_fichier_wav)        
        # Faire quelque chose avec le fichier .wav, par exemple :
        audio, current_sample_rate = sf.read(chemin_fichier_wav)



        # ecoacoustic indices
        # wave,fs = sound.load(filename=chemin_fichier_wav, channel='left', detrend=True, verbose=False)
        Sxx_power,tn,fn,ext = sound.spectrogram(audio, current_sample_rate, 
                                                 nperseg = 1024, noverlap=1024//2,
                                                 verbose = False, display = False,
                                                 savefig = None)
        S = -35         # Sensbility microphone-35dBV (SM4) / -18dBV (Audiomoth)
        G = 26+16       # Amplification gain (26dB (SM4 preamplifier))

        # compute all the audio indices and store them into a DataFrame
        # dB_threshold and rejectDuration are used to select audio events.
        # df_audio_ind = features.all_temporal_alpha_indices(audio, fs,
        #                                       gain = G, sensibility = S,
        #                                       dB_threshold = 3, rejectDuration = 0.01,
        #                                       verbose = False, display = False)
        

        df_spec_ind, df_spec_ind_per_bin = features.all_spectral_alpha_indices(Sxx_power,
                                                                tn, fn,
                                                                flim_low = [0,1500],
                                                                flim_mid = [1500,8000],
                                                                flim_hi  = [8000,20000],
                                                                gain = G, sensitivity = S,
                                                                verbose = False,
                                                                R_compatible = 'soundecology',
                                                                mask_param1 = 6,
                                                                mask_param2 = 0.5,
                                                                display = False)
        tabAcoustic_indices.append(df_spec_ind.to_numpy())

pickle.dump({'tabAcoustic_indices':tabAcoustic_indices}, open('./acoustic_indices.pkl', 'wb'))




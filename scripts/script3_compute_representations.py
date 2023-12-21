import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

def compute_stft(audio_path, output_folder):
    # Load audio file
    y, sr = librosa.load(audio_path,sr=8000)
    # Compute STFT
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y,hop_length=64,win_length=512)), ref=np.max)
    # D[D<.001]=0
    print(np.max(D))
    # Plot and save the STFT
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(os.path.join(output_folder, os.path.basename(audio_path) + '_stft.png'))
    plt.show()

def process_folder(folder_path, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.wav'):
            print(filename)
            audio_path = os.path.join(folder_path, filename)
            compute_stft(audio_path, output_folder)

if __name__ == "__main__":
    # Replace 'your_input_folder' and 'your_output_folder' with the actual folder paths
    input_folder = 'output/nn07c_20220603_000000/'
    output_folder = 'spectrogram'

    process_folder(input_folder, output_folder)

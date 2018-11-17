import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

sig, fs = librosa.load('snare.wav')
# make pictures name
save_path = 'test.jpg'


stft = np.abs(librosa.stft(sig))
spectrogram = np.mean(librosa.stft(sig).T, axis=0)
melspectrogram = np.mean(librosa.feature.melspectrogram(sig, sr=fs).T,axis=0)
mfccs = np.mean(librosa.feature.mfcc(y=sig, sr=fs, n_mfcc=40).T,axis=0)
spec_contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=fs).T,axis=0)
print("spectrogram ", spectrogram.shape)
print("melspectrogram ", melspectrogram.shape)
print("mfccs ", mfccs.shape)
print("spec_contrast ", spec_contrast.shape)

#image = librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
#print(image)

# plt.figure(figsize=(20, 4))
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
#plt.tight_layout()
#plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
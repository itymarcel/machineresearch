import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

sig, fs = librosa.load('turtles.m4a')
# make pictures name
save_path = 'test.jpg'


S = librosa.feature.melspectrogram(y=sig, sr=fs, n_mels=1024, fmax=16000)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

# plt.figure(figsize=(20, 4))
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
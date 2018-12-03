import os
import glob
import matplotlib.pyplot as plt
from time import time
from matplotlib.pyplot import specgram
import librosa
import librosa.display
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import json
from tempfile import TemporaryFile
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, load_model

from keras import activations
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.layers import Conv2D, MaxPooling1D, Conv1D, Flatten, LSTM
from keras.callbacks import TensorBoard
from sklearn.utils import class_weight

from vis.visualization import visualize_activation
from vis.utils import utils


def extract_feature_from_section(file_name, offset, duration):
    X, sample_rate = librosa.load(file_name, offset=offset, duration=duration)
    stft = np.abs(librosa.stft(X))
    #spectrogram = np.mean(librosa.stft(X).T, axis=0)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    #chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mel, contrast, mfccs
    #return mfccs,chroma,mel,contrast,tonnetz

def parse_prediction_file(file_name):
    X, sample_rate = librosa.load(file_name)
    duration = librosa.get_duration(y=X, sr=sample_rate)
    onset_env = librosa.onset.onset_strength(X, sr=sample_rate)
    #tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)
    tempo = 120
    offset = 0.0
    snapshot_duration = 60 / tempo / 4
    print('TEMPO: ', tempo)
    print('SNAP DURATION: ', snapshot_duration)
    features = np.empty([0,175])
    while offset < (duration-snapshot_duration):
      try:
        mel, contrast, mfccs = extract_feature_from_section(file_name, offset, snapshot_duration)
        ext_features = np.hstack([mel, contrast, mfccs])
        features = np.vstack([features,ext_features])
        offset += snapshot_duration
      except Exception as e:
        print(offset, e)
        ##os.remove(fn)
        continue
    return np.array(features)

prediction_file_features = parse_prediction_file('sound/simple_beat-with-synth.wav')
# simple_beat has 64 hits
prediction_file_features = np.expand_dims(prediction_file_features, axis=2)
model = load_model('./sound-prediction.h5')

#plt.rcParams['figure.figsize'] = (18, 6)
#layer_idx = utils.find_layer_idx(model, 'preds')
#model.layers[layer_idx].activation = activations.linear
#model = utils.apply_modifications(model)
#filter_idx = 0
#img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
#plt.imshow(img[..., 0])

y_proba = model.predict(prediction_file_features)
y_proba_percentage = model.predict(prediction_file_features)
y_probability = model.predict_classes(prediction_file_features)
y_classes = y_proba.argmax(axis=-1)
print('Leng: ', len(y_classes))
print('Pred: ', y_classes)
print('Perc: ', np.around(y_proba, decimals=3))

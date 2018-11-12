import os
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import librosa.display
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    #mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    #chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    #contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mel
    #return mfccs,chroma,mel,contrast,tonnetz


def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty([0,128]), np.empty(0) #128 is the length of the mel-spectrogram data array
    for label, sub_dir in enumerate(sub_dirs):
      for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
              mel = extract_feature(fn)
              ext_features = np.hstack([mel])
              features = np.vstack([features,ext_features])
              labels = np.append(labels, fn.split('/')[-1].split('-')[0])
              #print(labels)
              pass
            except:
              print(fn)
              #os.remove(fn)
              continue
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels) # length of lapels
    n_unique_labels = len(np.unique(labels)) #number of unique labels
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def return_number_labels(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    print("Number Unique Labels")
    print(n_unique_labels)
    return n_unique_labels


parent_dir = 'sound'
tr_sub_dirs = ['tr']
ts_sub_dirs = ['ts']
tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)

#tr_labels = one_hot_encode(tr_labels)
#ts_labels = one_hot_encode(ts_labels)

n_dim = tr_features.shape[1]
n_classes = return_number_labels(tr_labels)

print(ts_labels)
yr_binary = keras.utils.to_categorical(tr_labels)
ys_binary = keras.utils.to_categorical(ts_labels)

# KERAS
model = Sequential()

model.add(Dense(units=4, activation='relu', input_dim=n_dim))
model.add(Dense(units=2, activation='sigmoid'))

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



model.fit(tr_features, yr_binary, epochs=500, batch_size=32)
loss_and_metrics = model.evaluate(ts_features, ys_binary, batch_size=32)
y_proba = model.predict(ts_features)
y_classes = y_proba.argmax(axis=-1)
print('Pred: ', y_classes)
print('Labe: ', ts_labels)
# 0 - snare
# 1 - kick
# 2 - hihat
# 3 - bongo
# 4 - claps
# 5 - tambourine
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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.layers import Conv2D, MaxPooling1D, Conv1D, Flatten, LSTM
from keras.callbacks import TensorBoard
from sklearn.utils import class_weight
from vis.visualization import visualize_activation
from vis.utils import utils

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    segment = librosa.effects.trim(y=X, top_db=10, frame_length=1024, hop_length=256)
    audio_time_series = segment[0]
    stft = np.abs(librosa.stft(audio_time_series))
    #spectrogram = np.mean(librosa.stft(X).T, axis=0)
    mfccs = np.mean(librosa.feature.mfcc(y=audio_time_series, sr=sample_rate, n_mfcc=40).T,axis=0)
    #chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    #mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    mel_original = librosa.feature.melspectrogram(audio_time_series, sr=sample_rate).T
    mel = np.mean(mel_original, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mel, contrast, mfccs
    #return mfccs,chroma,mel,contrast,tonnetz


def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty([0,175]), np.empty(0) #128 is the length of the mel-spectrogram data array
    for label, sub_dir in enumerate(sub_dirs):
      for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
              mel, contrast, mfccs = extract_feature(fn)
              ext_features = np.hstack([mel, contrast, mfccs])
              print("EXT FEATURES shape: ", ext_features.shape)
              features = np.vstack([features,ext_features])
              labels = np.append(labels, fn.split('/')[-1].split('-')[0])
              #print(labels)
              pass
            except Exception as e:
              print(fn, e)
              #os.remove(fn)
              continue
    return np.array(features), np.array(labels, dtype = np.int)

def return_number_labels(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    print("Number Unique Labels")
    print(n_unique_labels)
    return n_unique_labels


parent_dir = 'sound'
tr_sub_dirs = ['tr']
ts_sub_dirs = ['ts']


try:
  tr_features = np.load('tr_features.npy')
  tr_labels = np.load('tr_labels.npy')
  ts_features = np.load('ts_features.npy')
  ts_labels = np.load('ts_labels.npy')
  print("All Numpy Arrrays successfully restored")
except:
  print("One or more arrays not found, therefor recreating them.")
  tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
  ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)
  np.save('tr_features.npy', tr_features)
  np.save('tr_labels.npy', tr_labels)
  np.save('ts_features.npy', ts_features)
  np.save('ts_labels.npy', ts_labels)



######################### split a song into fragments ###########################
# def parse_and_split_song(parent_dir,sub_dirs,file_ext='*.wav'):
##  librosa.effects.split

n_dim = tr_features.shape[1]
n_classes = return_number_labels(tr_labels)
print('N DIM : ', n_dim)
print('N Shape: ', tr_features.shape)
tr_labels_binary = keras.utils.to_categorical(tr_labels)
ts_labels_binary = keras.utils.to_categorical(ts_labels)

# safe test dataset to JSON formatted text
#with open('data.txt', 'w') as f:
#  json.dump(ts_features.tolist(), f, ensure_ascii=False)
#exit(0)

# KERAS
model = Sequential()

# this is just a NN, not a Convolution n_unique_labels
tr_features = np.expand_dims(tr_features, axis=2)
ts_features = np.expand_dims(ts_features, axis=2)
## ADDING A Long Short Term Memory (RNN?) Layer with 8 neurons)
## ADDING two CNN Layers
model.add(Conv1D(8, kernel_size=16, activation="relu", input_shape=(175, 1)))
#model.add(MaxPooling1D(3))
model.add(Conv1D(16, kernel_size=8, activation="relu"))
#model.add(MaxPooling1D(3))
model.add(Conv1D(32, kernel_size=4, activation="relu"))
#model.add(MaxPooling1D(3))
#model.add(Conv1D(64, kernel_size=4, activation="relu"))
#model.add(MaxPooling1D(3))
#model.add(LSTM(256, return_sequences=True, input_shape=(175,1)))
#model.add(Dropout(0.1))
#model.add(LSTM(512, return_sequences=True))
#model.add(Dropout(0.1))
model.add(Flatten())
#model.add(Dense(256))
#model.add(Flatten())
#model.add(Dense(units=128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(units=128, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(units=n_classes, activation='softmax', name='preds'))

# adam optimizer worked best so far
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=0,
          write_graph=True, write_images=True)

# set class weights because of imbalance (more files for kick, less for bongos)
class_weights_array = class_weight.compute_class_weight('balanced',
                                                 np.unique(tr_labels),
                                                 tr_labels)

class_weights = dict(enumerate(class_weights_array))
print("CLASS WEIGHTS ", class_weights)

model.fit(tr_features, tr_labels_binary, epochs=50, batch_size=32, class_weight=class_weights, callbacks=[tensorboard])
model.save('sound-prediction.h5')
#tfjs.converters.save_keras_model(model, 'tfjs-model/')
loss_and_metrics = model.evaluate(ts_features, ts_labels_binary, batch_size=32)
y_proba = model.predict(ts_features)
y_classes = y_proba.argmax(axis=-1)
correct_predictions = 0
print('Pred: ', y_classes)
print('Labe: ', ts_labels)
for index in range(len(y_classes)):
  if y_classes[index] == ts_labels[index]:
    correct_predictions += 1

prediction_accuracy = correct_predictions/len(y_classes)

print('prediction accuracy: ', prediction_accuracy)
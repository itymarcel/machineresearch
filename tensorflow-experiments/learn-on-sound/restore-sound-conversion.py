import os
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import librosa.display
import tensorflow as tf
import numpy as np
import pandas as pd


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)

    return mfccs,chroma,mel,contrast,tonnetz


def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty([0,193]), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
      for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
              ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
              features = np.vstack([features,ext_features])
              labels = np.append(labels, fn.split('/')[-1].split('-')[0])
              #print(labels)
              pass
            except:
              print(fn)
              os.remove(fn)
              continue
    return np.array(features), np.array(labels, dtype = np.int)

parent_dir = 'sound'
ts_sub_dirs = ['ts']
ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)

# Create some variables.
#v1 = tf.get_variable("v1", shape=[3])
#v2 = tf.get_variable("v2", shape=[5])
n_dim = ts_features.shape[1]
n_classes = 2
X = tf.placeholder(tf.float32,[None,n_dim], name="X")
Y = tf.placeholder(tf.float32,[None,n_classes], name="Y")

tf.reset_default_graph()
saver = tf.train.import_meta_graph('./kicksnare.ckpt.meta')

with tf.Session() as sess:
  # saver.restore(sess,tf.train.latest_checkpoint('./'))
  saver.restore(sess, './kicksnare.ckpt')
  print("Model restored.")
  # W_1 = sess.run('W_1:0')
  #y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)
  y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
  #print('Prediction: ', y_pred)
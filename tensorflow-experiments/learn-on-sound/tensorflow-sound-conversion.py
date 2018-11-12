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

tr_labels = one_hot_encode(tr_labels)
ts_labels = one_hot_encode(ts_labels)

training_epochs = 5000
n_dim = tr_features.shape[1]
n_classes = return_number_labels(tr_labels)
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

X = tf.placeholder(tf.float32,[None,n_dim], name="X")
Y = tf.placeholder(tf.float32,[None,n_classes], name="Y")

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd), name="W_1")
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd), name="b_1")
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)


W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd), name="W_2")
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd), name="b_2")
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2, name="h_2")


W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd), name="W")
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd), name="b")
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b, name="y_")

init = tf.initialize_all_variables()
saver = tf.train.Saver()

cost_function = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None
print("tr features shape")
print(tr_features.shape)
print("tr labels shape")
print(tr_labels.shape)
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
        cost_history = np.append(cost_history,cost)

    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
    print('prediction: ', y_pred);
    y_true = sess.run(tf.argmax(ts_labels,1))
    print('Test accuracy: ',round(sess.run(accuracy, feed_dict={X: ts_features, Y: ts_labels}), 3))
    save_path = saver.save(sess, "./kicksnare.ckpt")
    print("Model saved in path: %s" % save_path)

#fig = plt.figure(figsize=(10,8))
#plt.plot(cost_history)
#print(cost_history)
#plt.axis([0,training_epochs,0,np.max(cost_history)])
#plt.show()

#p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
#print("F-Score:", round(f,3))
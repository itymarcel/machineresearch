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
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import TensorBoard

try:
  np.load('fnumpy.npy')
  print("file found")
except:
  print("file not found")
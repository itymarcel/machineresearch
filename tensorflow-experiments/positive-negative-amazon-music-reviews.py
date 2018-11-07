import tensorflow as tf
from tensorflow import keras

import numpy as np


train_data = keras.preprocessing.sequence.pad_sequences(train_data,
  value=word_index["<PAD>"],
  padding='post',
  maxlen=256)


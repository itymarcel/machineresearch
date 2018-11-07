import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from tensorflow import keras


train_data = keras.preprocessing.sequence.pad_sequences(train_data,
  value=word_index["<PAD>"],
  padding='post',
  maxlen=256)


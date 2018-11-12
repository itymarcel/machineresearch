import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

amazon = pd.read_json('amazon-music-reviews-short.json', lines=True)
amazon['reviewText length'] = amazon['reviewText'].apply(len)
amazon.head()

amazon_tensor = tf.io.decode_json_example(
    'amazon-music-reviews-short.json',
    name=None
)
# amazon_numpy = pd.read_json('amazon-music-reviews-short.json', orient="index").applymap(np.array)
dict = json.loads('amazon-music-reviews-short.json')
numpy_2d_arrays = [np.array(ring) for ring in dict["rings"]]
print(numpy_2d_arrays)

# g = sns.FacetGrid(data=amazon, col='overall')
# g.map(plt.hist, 'reviewText length', bins=50)
# sns.boxplot(x='overall', y='reviewText length', data=amazon)
# stars = amazon.groupby('overall').mean()
# sns.heatmap(data=stars.corr(), annot=True)
# plt.show()
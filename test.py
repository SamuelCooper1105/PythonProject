import tensorflow as tf
from tensorflow import keras
import numpy as np

# Okay so lets actually laod the data here. 
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=10000)

#okay so lets print an example review as tokeniezed numbers
print("First training example (tokenized):", train_data[0])

print("Label:", train_labels[0])

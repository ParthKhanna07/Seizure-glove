from keras.regularizers import l2,l1
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import lite

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# Importing the dataset
dataset = pd.read_csv('final epilepsy.csv')
X = dataset.iloc[:, [2, 3,4,5]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(12, input_dim=4, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=150, batch_size=100)
# evaluate the keras model
keras_file="Epileptic.h5"
tf.keras.models.save_model(model, keras_file)
_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))
predictions = model.predict_classes(X_test)



converter = lite.TFLiteConverter.from_keras_model_file('Epileptic.h5')

tflite_model = converter.convert()
open("Epileptic_final.tflite","wb").write(tflite_model)
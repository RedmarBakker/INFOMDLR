# Assignment 1
#
# a) Select your choice of neural networks model that is suitable for this task and motivate it. Train your model
# to predict one step ahead data point, during training (see following Figure). Scale your data before training
# and scale them back to be able to compare your predictions with real measurements
#
# b) How many past time steps should you input into your network to achieve the best possible performance?
# (Hint: This is a tunable parameter and needs to be tuned).
#
# c) Once your model is trained, use it to predict the next 200 data points recursively. This means feeding each
# prediction back into the model to generate the subsequent predictions.)
#
# d) On May 9th, download the real test dataset and evaluate your model by reporting both the Mean Absolute
# Error (MAE) and Mean Squared Error (MSE) between its predictions and the actual test values. Additionally,
# create a plot comparing the predicted and real values to visually assess your model’s performance.

import scipy.io
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from helpers import create_training_data

# load the training data
dataset = scipy.io.loadmat('Xtrain.mat')

history_length = 2
batch_size = 64

X = np.array(dataset['Xtrain'])

train_size = int(len(X) * 0.8)
train, test = X[:train_size], X[train_size:]

x_train, y_train = create_training_data(train, history_length)
x_test, y_test = create_training_data(test, history_length)

# example code
model = keras.Sequential([
    layers.LSTM(128, input_shape=(1, 1)),
    layers.Dense(10),
])

model.summary()

model.compile(
    loss='mse',
    optimizer="sgd",
    metrics=["accuracy"],
)

model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=100
)
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
mat = scipy.io.loadmat('Xtrain.mat')

history_length = 2

data = np.array(mat)
train = create_training_data(data, history_length + 1)


print(train)
exit()

# example code
model = keras.Sequential()

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))

model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)

model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=1
)
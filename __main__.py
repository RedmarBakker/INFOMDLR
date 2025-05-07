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
from helpers import create_training_data, scale_data
import matplotlib.pyplot as plt

# load the training data
dataset = scipy.io.loadmat('Xtrain.mat')

history_length = 10
batch_size = 64
train_test_balance = 0.8
epochs = 100

X = np.array(dataset['Xtrain'])

plt.plot(X)
plt.show()

min_val = np.min(X)
max_val = np.max(X)

print('min: ', min_val, 'max: ', max_val)

X = scale_data(X)

train_size = int(len(X) * train_test_balance)
train, validation = X[:train_size], X[train_size:]

x_train, y_train = create_training_data(train, history_length)
x_val, y_val = create_training_data(validation, history_length)

# example code
model = keras.Sequential([
    layers.Input(shape=(history_length, 1)),
    layers.LSTM(32),
    layers.Dense(1, activation='relu'),
])

model.summary()

model.compile(
    loss='mse',
    optimizer="adam",
    metrics=["accuracy"],
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

## show loss results of training

# Extract accuracies
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs+1), train_loss, label='Training loss', marker='o')
plt.plot(range(1, epochs+1), val_loss, label='Test loss', marker='s')
plt.title('Training and Test Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, epochs+1))
plt.ylim(0, 0.1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

## predict into the future

current = scale_data(X[:history_length])

step = 0
predictions = []
n_step_to_predict = 200
while step < n_step_to_predict:
    prediction = model.predict([current])[0]

    predictions.append(prediction)
    np.append(current, prediction)
    current = current[:history_length]

    step += 1

plt.plot(predictions)
plt.show()

# print(model.predict([
#     [1,2,3,4,5,6,7,8,9,10],
#     [10,9,8,7,6,5,4,3,2,1],
#     [12,15,19,20,30,60,100,160,120],
# ]))
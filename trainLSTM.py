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
import json
import tensorflow as tf
import keras
from keras import layers
from helpers import create_training_data, scale_data, scale_x_axis, generate_sequence
import matplotlib
# Use the TkAgg backend for matplotlib to avoid the need for a display
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# load the training data
dataset = scipy.io.loadmat('Xtrain.mat')

#Hyperparameters
# history_length = 50
batch_size = 64
train_test_balance = 0.8
epochs = 100

#Normalize and split the data
X = np.array(dataset['Xtrain'])
X = scale_data(X)

# time_steps = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
time_steps = [50,100]
for step in time_steps:
    history_length = step

    train_size = int(len(X) * train_test_balance)
    train, validation = X[:train_size], X[train_size:]

    x_train, y_train = create_training_data(train, history_length)
    x_val, y_val = create_training_data(validation, history_length)

    # Reshape data to have shape (n_samples, history_length, 1)
    x_train = x_train.reshape((x_train.shape[0], history_length,1))
    x_val = x_val.reshape((x_val.shape[0], history_length,1))

    # Define the model
    # Model heuristic:
    # 1. # of layers = # of features in the data
    # 2. # of units (neurons) = # of timesteps (history_length)
    model = keras.Sequential([
        layers.LSTM(units=120, input_shape=(history_length, 1)),
        layers.Dense(20, activation='relu'),
        layers.Dense(1, activation='linear'),
    ])

    #Create the model
    model.compile(
        loss='mse',
        optimizer="adam",
        metrics=["mean_squared_error"],
    )

    #Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs
    )

    # Extract loss values from the training history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Max accuracies for scaling the plots
    max_train_loss = np.max(train_loss)
    max_val_loss = np.max(val_loss)

    #Lowest loss in training and validation
    best_train_loss = np.min(train_loss)
    best_val_loss = np.min(val_loss)
    loss_data = np.array([best_train_loss, best_val_loss])

    #Save the model and the associated loss
    model.save(f"./models/LSTM{history_length}.keras")
    with open(f"./loss/LSTM{history_length}.json", "w") as f:
        json.dump(loss_data.tolist(), f)

#Generate 200 predictions points
# data_points = 200
# last_sequence = validation[-history_length:].reshape((1, history_length, 1))
# predicted_sequence = generate_sequence(last_sequence, model, history_length, num_steps=data_points)
# plt.figure(figsize=(8, 5))
# plt.plot(range(0, data_points), predicted_sequence, label='Predicted', marker='o')
# plt.title('Predicted Data Points')
# plt.xlabel('Data Point Index')
# plt.ylabel('Value')
# plt.xticks(range(0, data_points, 10))
# plt.ylim(0, 255)
# plt.grid(True)
# plt.show()

# Plotting the training and validation loss
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, epochs+1), train_loss, label='Training loss', marker='o')
# plt.plot(range(1, epochs+1), val_loss, label='Test loss', marker='s')
# plt.title('Training and Test Loss per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.xticks(scale_x_axis(epochs))
# plt.ylim(0, max(max_train_loss, max_val_loss) * 1.1)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


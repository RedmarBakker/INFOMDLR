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

# load the training data
mat = scipy.io.loadmat('Xtrain.mat')






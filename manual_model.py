import scipy.io
import numpy as np
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt

# load the training data
dataset = scipy.io.loadmat('Xtrain.mat')
X = np.array(dataset['Xtrain'])

batch_size = 64
history_size = 20 # still need to tweak this!
train_test_balance = 0.8

def create_data(raw_dat, time_steps):
    histories, labels = [], []

    for i in range(len(raw_dat) - time_steps):
        history = raw_dat[i:i+time_steps]
        label = raw_dat[i+time_steps]

        histories.append(history)
        labels.append(label)

    return np.array(histories), np.array(labels)

train_size = int(len(X) * train_test_balance)
val_size = int(len(X) * (1-train_test_balance))
train, validation = X[val_size:], X[:val_size] # the latter part of the data seems to be easier to predict for the model / lower error
                                               # we should have a better way to randomize the split, probably for tuning hyperparams too
                                               # I think it s because some parts of the data just have smaller values? which makes MSE a difficult comparison
# train, validation = X[:train_size], X[train_size:]
x_train, y_train = create_data(train, history_size)
x_val, y_val = create_data(validation, history_size)

# batch and shuffle the training data
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0])
batched_dataset = dataset.batch(batch_size)

#
# create model
class LaserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.recurrent = layers.SimpleRNN(120, activation='relu')
        self.dense = layers.Dense(120, activation='relu')
        self.output_layer = layers.Dense(1)

    def call(self, x, training=None):
        out = self.recurrent(x)
        out = self.dense(out)
        out = self.output_layer(out)

        return out
    
#
# define model
mdl = LaserModel()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
val_loss = tf.keras.metrics.Mean(name='val_loss')

#
# define steps
def train_step(trainx, trainy):
    # forward
    with tf.GradientTape() as tape:
        y_pred = mdl.call(trainx, training=True)
        loss = tf.keras.losses.MSE(trainy, y_pred)

    # update weights
    grads = tape.gradient(loss, mdl.trainable_variables)
    optimizer.apply_gradients(zip(grads, mdl.trainable_variables))

    # update loss/accuracy
    train_loss(loss)

def val_step(valx, valy):
    # forward
    y_pred = mdl.call(valx)
    loss = tf.keras.losses.MSE(valy, y_pred)

    # update loss/accuracy
    val_loss(loss)

#
# training
epochs = 100
train_losses = []
val_losses = []
for i in range(epochs):
    # train
    train_loss.reset_state()
    val_loss.reset_state()
    for x, y in batched_dataset:
        train_step(x, y)
    
    # val
    val_step(x_val, y_val)

    # print results
    print(f"epoch: {i} | train loss: {train_loss.result()} | val loss: {val_loss.result()}")
    train_losses.append(train_loss.result())
    val_losses.append(val_loss.result())


#
# visualize training arc
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2, marker='o')
plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2, marker='s')

# Customize the plot
plt.title('Training and Validation Loss Over Epochs', fontsize=14, pad=20)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust x-axis ticks to match epoch numbers
plt.xticks(np.arange(len(train_losses)), np.arange(1, len(train_losses)+1))

# Show the plot
plt.tight_layout()
plt.show()


#
# visualize ground truth against model's predictions
full_dat, _ = create_data(X, history_size)
model_dat = mdl.call(full_dat).numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) 
ax1.plot(X)
ax1.set_title('Original data')
ax2.plot(model_dat)
ax2.set_title('Model predictions')
plt.tight_layout()
plt.show()
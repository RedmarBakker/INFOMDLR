import scipy.io
import numpy as np
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt

# load the training data
dataset = scipy.io.loadmat('Xtrain.mat')
dat = np.array(dataset['Xtrain'])

batch_size = 64
history_size = 22
train_test_balance = 0.8

def create_data(raw_dat, time_steps):
    histories, labels = [], []

    for i in range(len(raw_dat) - time_steps):
        history = raw_dat[i:i+time_steps]
        label = raw_dat[i+time_steps]

        histories.append(history)
        labels.append(label)

    return np.array(histories), np.array(labels)

# zero-center & normalize
X_mean = np.mean(dat)
X_std = np.std(dat)
X = (dat - X_mean) / X_std

# create items+labels & shuffle them
np.random.seed(seed=11)
tf.random.set_seed(seed=11)
train_size = int(len(X) * train_test_balance)
x_all, y_all = create_data(X, history_size)
shuffled_idxs = np.random.permutation(len(x_all))
x_all, y_all = x_all[shuffled_idxs], y_all[shuffled_idxs]

# create train/val split
x_train, y_train = x_all[:train_size], y_all[:train_size]
x_val, y_val = x_all[train_size:], y_all[train_size:]

# batch the training data
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) #.shuffle(x_train.shape[0])
batched_dataset = dataset.batch(batch_size)

#
# create model
class LaserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.recurrent = layers.SimpleRNN(120, activation='relu', kernel_initializer='he_normal')
        # self.recurrent = layers.LSTM(256, activation='relu', kernel_initializer='he_normal')
        self.dense = layers.Dense(120, activation='relu', kernel_initializer='he_normal')
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
    loss = (tf.sqrt(loss) * X_std) # back to original units to make loss more interpretable
    train_loss(loss)

def val_step(valx, valy):
    # forward
    y_pred = mdl.call(valx)
    loss = tf.keras.losses.MSE(valy, y_pred)

    # update loss/accuracy
    loss = (tf.sqrt(loss) * X_std)
    val_loss(loss)

#
# training
epochs = 30
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
model_dat_scaled = (model_dat * X_std) + X_mean # scale data back

# visualize difference between the two
diff = dat[history_size:] - model_dat_scaled
# axs[1].plot(np.append([0]*history_size, diff))
plt.figure(figsize=(10, 6))
plt.title('Difference Between Ground Truth and Model Predictions', fontsize=14, pad=20)
plt.ylabel('Difference', fontsize=12)
plt.plot(np.append([0]*history_size, diff))
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


# 
# predict 200 points recursively

# load the test data
test_dataset = scipy.io.loadmat('Xtest.mat')
test_dat = np.array(test_dataset['Xtest'])

current = tf.reshape(full_dat[-1], [1,history_size,1]) # start at final point of training data
current = tf.cast(current, dtype=tf.float32)
result = []
for i in range(200):
    print(f"Predicting point {i+1}...")

    # save prediction
    pred = mdl.call(current)
    scaled_res = (tf.reshape(pred, [-1]).numpy() * X_std) + X_mean
    result.append(scaled_res)

    # update current
    current = current[:,1:,:] # trim off the first element
    current = tf.concat([current, tf.reshape(pred, [1,1,1])], axis=1) # append the prediction

test_loss_MSE = np.mean(np.square(test_dat - np.array(result)))
test_loss_MAE = np.mean(np.abs(test_dat - np.array(result)))
print("Evaluating test dataset...")
print(f"MSE: {test_loss_MSE} | MAE: {test_loss_MAE}")

plt.figure(figsize=(10, 6))
plt.title('Model Predictions vs. Actual Test Data', fontsize=14, pad=20)
plt.plot(test_dat, 'b', label='Test Data')
plt.plot(result, 'r--', label='Model Predictions')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import math
import json
import matplotlib
# Use the TkAgg backend for matplotlib to avoid the need for a display
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from data import load_files, DataGenerator, glue_chunks, CustomAccuracyLossCheckpoint
from models.spat_temp import build_spat_temp_model


# ---- TRAIN THE CNN ----
# Select dataset
model = 'intra' # or 'intra'
match model:
    case 'intra':
        filepath = './intra/train'
    case 'cross':
        filepath = './cross/train'
    case _:
        raise ValueError(f"Unknown training set: {model}")

# Train-test split
train_files, val_files = train_test_split(load_files(filepath),test_size=0.18,
                                          random_state=42)
# Select the autoencoder model
match model:
    case 'intra':
        model_path = './models/autoencoder_intra_dropout+L2.keras'
    case 'cross':
        model_path = './models/autoencoder_cross_dropout+L2.keras'
    case _:
        raise ValueError(f"Unknown model: {model}")
try:
    autoencoder = keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading the model from {model_path}: {e}")

encoder = autoencoder.get_layer('encoder')
print(f"Encoder loaded")

# Hyperparameter
# Encoder
ENCODER = 'encoder'
BATCH_SIZE_ENCODER = 32
CHUNK_SIZE = 128

# CNN
EPOCHS = 100
BATCH_SIZE = 32
DOWNSAMPLE_FREQ = 500

# Prepare the data
print("Creating latent representations for data...")
EncoderTrainData = DataGenerator(filepaths=train_files,
                                         batch_size=BATCH_SIZE_ENCODER,
                                         task_type=ENCODER,
                                         chunk_size=CHUNK_SIZE,
                                         shuffle=False)
EncoderValData = DataGenerator(filepaths=val_files,
                                         batch_size=BATCH_SIZE_ENCODER,
                                         task_type=ENCODER,
                                         chunk_size=CHUNK_SIZE,
                                         shuffle=False)

print("Collecting data and labels for encoder...")

all_X_train_batches = []
all_y_train_batches = []

for i in range(len(EncoderTrainData)):  # Iterate through all batches
    X_batch, y_batch = EncoderTrainData[i]  # Get X and Y for each batch
    if X_batch.shape[0] > 0:  # Only append if the batch is not empty
        all_X_train_batches.append(X_batch)
        all_y_train_batches.append(y_batch)
    else:
        print(f"WARNING: Skipping empty batch {i} from encoder_train_generator during collection.")

X_train_full_dataset = np.concatenate(all_X_train_batches, axis=0)
y_train_full_dataset = np.concatenate(all_y_train_batches, axis=0)

all_X_val_batches = []
all_y_val_batches = []
for i in range(len(EncoderValData)):
    X_batch_val, y_batch_val = EncoderValData[i]
    if X_batch_val.shape[0] > 0:  # Only append if the batch is not empty
        all_X_val_batches.append(X_batch_val)
        all_y_val_batches.append(y_batch_val)
    else:
        print(f"WARNING: Skipping empty batch {i} from encoder_val_generator during collection.")

X_val_full_dataset = np.concatenate(all_X_val_batches, axis=0)
y_val_full_dataset = np.concatenate(all_y_val_batches, axis=0)


# Concatenate the labels
nr_chunks = int(math.floor(35624 / CHUNK_SIZE))

y_train = y_train_full_dataset[::nr_chunks]
y_val = y_val_full_dataset[::nr_chunks]

print(f"train data shape: {X_train_full_dataset.shape}")
print(f"train labels shape: {y_train.shape}")
print(f"val data shape: {X_val_full_dataset.shape}")
print(f"val labels shape: {y_val.shape}")

# Create the latent representations and glue them for training
X_train_latent_chunks = encoder.predict(EncoderTrainData)
X_val_latent_chunks = encoder.predict(EncoderValData)

X_train_latent = glue_chunks(X_train_latent_chunks, nr_chunks=nr_chunks)
X_val_latent = glue_chunks(X_val_latent_chunks, nr_chunks=nr_chunks)

input_shape = X_train_latent[0].shape

# ---- BUILD THE CNN ----
EMBED_DIM = 14
cnn = build_spat_temp_model(input_shape, embed_dim=EMBED_DIM, dropout=0.5, l2fac=0.004)
# intra embed=14, dropout=0.5, l2fac=0.004, 2 residuals
# cross embed=14, dropout=0.5, l2fac=0.004, 2 residuals

# ---- Callbacks ----
ReducedLR = keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=int(EPOCHS/10),
    min_lr=1e-6,
    verbose=1
)

EarlySaving = CustomAccuracyLossCheckpoint(
    filepath=f'models/fixed _cnn_{model}_filters_{EMBED_DIM}_residual_2_model.keras',
    verbose=1
)

callbacks = [EarlySaving]

history = cnn.fit(
    X_train_latent, y_train,
    validation_data=(X_val_latent, y_val),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

# Create a figure and a set of subplots.
fig, ax1 = plt.subplots(figsize=(10, 6))

# Find minimum validation loss and maximum validation accuracy epochs
best_val_loss_epoch = np.argmin(history.history['val_loss'])
best_val_accuracy_epoch = np.argmax(history.history['val_accuracy'])

# Get the actual best values
min_val_loss = history.history['val_loss'][best_val_loss_epoch]
max_val_accuracy = history.history['val_accuracy'][best_val_accuracy_epoch]

# Cast to int and float for the json.dump()
best_val_loss_epoch = int(best_val_loss_epoch)
best_val_accuracy_epoch = int(best_val_accuracy_epoch)
min_val_loss = float(min_val_loss)
max_val_accuracy = float(max_val_accuracy)

# Save the validation loss and validation accuracy to json file
validation_history = {
    'val_loss (epoch)': (min_val_loss, best_val_loss_epoch + 1),
    'val_accuracy (epoch)': (max_val_accuracy, best_val_accuracy_epoch + 1),
    'model': model
}

with open(f'./models/cnn_history/fixed_autoencoder_{model}_filters_{EMBED_DIM}.json', 'w') as f:
    json.dump(validation_history, f, indent=4)

# --- Plot Loss on the left Y-axis ---
color = 'tab:red'
ax1.set_ylabel('Loss', color=color) #
ax1.plot(history.history['loss'], label='Training Loss', color=color, linestyle='-')
ax1.plot(history.history['val_loss'], label='Validation Loss', color=color, linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)

ax1.axvline(x=best_val_loss_epoch, color='orange', linestyle=':', linewidth=1.5,
            label=f'Best Val Loss ({min_val_loss:.4f} at E{best_val_loss_epoch + 1})')

# --- Create a second Y-axis for Accuracy ---
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(history.history['accuracy'], label='Training Accuracy', color=color, linestyle='-')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color=color, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

ax2.axvline(x=best_val_accuracy_epoch, color='green', linestyle=':', linewidth=1.5,
            label=f'Best Val Accuracy ({max_val_accuracy:.4f} at E{best_val_accuracy_epoch + 1})')

# --- Add Title and Legend ---
plt.title(f'CNN Training History with Autoencoder ({model}-subject)')

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right')

plt.grid(True)
fig.tight_layout()
plt.show()
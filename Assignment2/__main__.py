# (a) Choose a suitable deep learning model for the involved classification tasks. Justify your choice.
# (b) Compare the accuracy of the 2 types of classification, i.e. intra-subject and cross subject data using your model. Explain your results.
# (c) Explain the choices of hyper-parameters of your model architecture and analyze their influence on the results (for both 2 types of classification). How they are selected?
# (d) If there is a significant difference in training and testing accuracies, what could be a possible reason? What are the alternative models or approaches you would select? Select one and implement to further improve your results. Justify your choice.
import numpy as np
from sklearn.model_selection import train_test_split
import os
import keras
import math
import json
import matplotlib
from itertools import chain
# Use the TkAgg backend for matplotlib to avoid the need for a display
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from data import z_norm, load, load_files, extract_patches, DataGenerator, glue_chunks, CustomAccuracyLossCheckpoint
from visualization import plot_dataset_as_meg, plot_dataset_as_lines
from models.autoencoder import build_autoencoder, chunk_data, chunk_data_for_conv2d
from models.transformer import build_transformer
from models.spat_temp import build_spat_temp_model


# plot_dataset_as_lines(z_norm(load(rest_set_names[0])))
# plot_dataset_as_meg(load(rest_set_names[0]))
# plot_dataset_as_meg(load(motor_set_names[0]))
# plot_dataset_as_meg(load(math_set_names[0]))
# plot_dataset_as_meg(load(memory_set_names[0]))



# ---- TRAINING ----
# filepaths = ["./Intra/train", "./Intra/test", "./Cross/train",
#              "./Cross/test1", "./Cross/test2", "./Cross/test3"]
# train_files = []
# test_files = []
# for i, filepath in enumerate(filepaths):
#     loaded = load_files(filepath)
#
#     if 'train' in filepath:
#         train_files.append(loaded)
#     elif 'test' in filepath:
#         test_files.append(loaded)
#
#
# train_flattened = list(chain.from_iterable(train_files))
# test_flattened = list(chain.from_iterable(test_files))
#
# all_files = train_flattened + test_flattened
#
# # print(f"Nr of training files: {len(train_flattened)}")
# # print(f"Nr of testing files: {len(test_flattened)}")
# # print(f"Total number of files: {len(train_flattened) + len(test_flattened)}")
#
# # Apply z-norm to the
# for subject in all_files:
#     data = load(subject)
#     highest = -np.inf
#     lowest = np.inf
#
#     #normalised = z_norm(data)
#
#     for channel in data:
#         if np.max(channel) > highest:
#             highest = np.max(channel)
#         if np.min(channel) < lowest:
#             lowest = np.min(channel)
#
# print(f"Highest value: {highest}\n"
#       f"Lowest value: {lowest}")

# ---- TRAINING SETUP ----
# hyperparameters
EPOCHS = 100
BATCH_SIZE = 32

# Process data
data = 'intra' # or 'cross'
match data:
    case 'intra':
        filepath = "./Intra/train"
    case 'cross':
        filepath = "./Cross/train"
    case _:
        raise ValueError(f"Unknown data type: {data}")

# Collect all training files
all_files = []
for root, _, files in os.walk(filepath):
    for file in files:
        if file.endswith('.h5'):
            all_files.append(os.path.join(root, file))
all_files.sort()

if not all_files:
    raise ValueError(f"No .h5 files found in {dir}. Check path and file extensions.")

# Train test split
train_filepaths, val_filepaths = train_test_split(
    all_files,
    test_size=0.18,
    random_state=42)
print(len(train_filepaths))
train_generator = DataGenerator(
        filepaths=train_filepaths,
        batch_size=BATCH_SIZE,
        task_type='classifier',
        shuffle=True)
val_generator = DataGenerator(
        filepaths=val_filepaths,
        batch_size=BATCH_SIZE,
        task_type='classifier',
        shuffle=True)
print(f"Split!")
print(len(train_generator))

# Build the Transformer model
MegDataShape = (248, 35624)
config = {}
config['num_layers'] = 4
config['embedding_size'] = 16
config['num_heads'] = 4
config['dropout_rate'] = 0.1
config['num_channels'] = 1
config['mlp_dim'] = 64
config['patch_size'] = MegDataShape[0]
config['num_patches'] = math.floor((config['patch_size'] * MegDataShape[1]) / (config['patch_size'] * config['patch_size']))

# Transform data into patches
for i in range(len(train_generator)):
    X_batch, y_batch = train_generator[i]
    train_input = np.array([extract_patches(sample, (config['patch_size'], config['patch_size']), config['num_patches']) for sample in X_batch])
    train_labels = np.array(y_batch)

for i in range(len(val_generator)):
    X_batch, y_batch = val_generator[i]
    val_input = np.array([extract_patches(sample, (config['patch_size'], config['patch_size']), config['num_patches']) for sample in X_batch])
    val_labels = np.array(y_batch)

# Build the transformer
transformer = build_transformer(config)
transformer.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# Callbacks
ModelSaving = keras.callbacks.ModelCheckpoint(
    filepath='./models/transformer.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
callbacks = [ModelSaving]
# Fit the model
history = transformer.fit(
    train_input, train_labels,
    validation_data=(val_input, val_labels),
    epochs=EPOCHS,
    callbacks=callbacks,
)
# Plot the accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Transformer Model Accuracy')
plt.legend()
plt.grid(True)
plt.show()



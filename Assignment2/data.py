# (a) Choose a suitable deep learning model for the involved classification tasks. Justify your choice.
# (b) Compare the accuracy of the 2 types of classification, i.e. intra-subject and cross subject data using your model. Explain your results.
# (c) Explain the choices of hyper-parameters of your model architecture and analyze their influence on the results (for both 2 types of classification). How they are selected?
# (d) If there is a significant difference in training and testing accuracies, what could be a possible reason? What are the alternative models or approaches you would select? Select one and implement to further improve your results. Justify your choice.

import os
import glob
import math
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mne
from matplotlib.animation import FuncAnimation

# Build the cross train dataset
# Since the cross train dataset is too large to fit in memory, it has to be loaded in batches

# Data generator class for cross-subject training:
class CrossSubjectDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, filepaths,
                 chunk_size, batch_size,
                 shuffle=True, **kwargs
                 ):
        super().__init__(**kwargs)
        self.filepaths = filepaths
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        # This list will store references to individual chunks (file index, start/end time)
        self._all_chunk_references = []

        # Populate _all_chunk_references by inspecting each file's metadata
        for filepath_idx, filepath in enumerate(self.filepaths):
            try:
                # Use h5py to get shape without loading full data
                with h5py.File(filepath, 'r') as f:
                    data_name = list(f.keys())[0]  # Assumes one dataset per .h5 file
                    # Assuming data is (channels, total_measurements)
                    _, total_measurements = f[data_name].shape
            except Exception as e:
                print(f"Error reading metadata from {filepath}: {e}. Skipping file.")
                continue

            n_chunks_in_file = total_measurements // self.chunk_size
            for i in range(n_chunks_in_file):
                self._all_chunk_references.append({
                    'filepath_idx': filepath_idx,  # Index into self.filepaths list
                    'start_time_idx': i * self.chunk_size,
                    'end_time_idx': (i + 1) * self.chunk_size
                })

        #Shuffle all indices after each epoch to prevent overfitting
        self.on_epoch_end()

    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch
        self.indices = np.arange(len(self._all_chunk_references))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self._all_chunk_references) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch_chunks = []

        loaded_file_cache = {}  # Simple cache for this __getitem__ call

        for ref_idx in indices:
            ref = self._all_chunk_references[ref_idx]
            filepath = self.filepaths[ref['filepath_idx']]

            if filepath not in loaded_file_cache:
                file_data = load(filepath)
                # Ensure data is 2D (channels, time_steps) if load() returns (1, C, t)
                if file_data.ndim == 3 and file_data.shape[0] == 1:
                    file_data = file_data.squeeze(axis=0)
                znormdata = z_norm(file_data)
                loaded_file_cache[filepath] = znormdata

            full_data_from_file = loaded_file_cache[filepath]

            chunk = full_data_from_file[:, ref['start_time_idx']: ref['end_time_idx']]
            chunk_reshaped = chunk.reshape(chunk.shape[0], chunk.shape[1], 1)  # (num_channels, chunk_size, 1)
            X_batch_chunks.append(chunk_reshaped)

        X_batch = np.stack(X_batch_chunks)
        return X_batch, X_batch  # For autoencoder


# Build the intra train dataset
rest_set_names = [
    "./Intra/train/rest_105923_1.h5",
    "./Intra/train/rest_105923_2.h5",
    "./Intra/train/rest_105923_3.h5",
    "./Intra/train/rest_105923_4.h5",
    "./Intra/train/rest_105923_5.h5",
    "./Intra/train/rest_105923_6.h5",
    "./Intra/train/rest_105923_7.h5",
    "./Intra/train/rest_105923_8.h5",
]
motor_set_names = [
    "./Intra/train/task_motor_105923_1.h5",
    "./Intra/train/task_motor_105923_2.h5",
    "./Intra/train/task_motor_105923_3.h5",
    "./Intra/train/task_motor_105923_4.h5",
    "./Intra/train/task_motor_105923_5.h5",
    "./Intra/train/task_motor_105923_6.h5",
    "./Intra/train/task_motor_105923_7.h5",
    "./Intra/train/task_motor_105923_8.h5",
]
math_set_names = [
    "./Intra/train/task_story_math_105923_1.h5",
    "./Intra/train/task_story_math_105923_2.h5",
    "./Intra/train/task_story_math_105923_3.h5",
    "./Intra/train/task_story_math_105923_4.h5",
    "./Intra/train/task_story_math_105923_5.h5",
    "./Intra/train/task_story_math_105923_6.h5",
    "./Intra/train/task_story_math_105923_7.h5",
    "./Intra/train/task_story_math_105923_8.h5",
]
memory_set_names = [
    "./Intra/train/task_working_memory_105923_1.h5",
    "./Intra/train/task_working_memory_105923_2.h5",
    "./Intra/train/task_working_memory_105923_3.h5",
    "./Intra/train/task_working_memory_105923_4.h5",
    "./Intra/train/task_working_memory_105923_5.h5",
    "./Intra/train/task_working_memory_105923_6.h5",
    "./Intra/train/task_working_memory_105923_7.h5",
    "./Intra/train/task_working_memory_105923_8.h5",
]
intraTrain = [rest_set_names, math_set_names, motor_set_names, memory_set_names]
def load(filename):
    file = h5py.File(filename, 'r')

    for name in file:
        return file.get(name)[()]

def build_intra_dataset():
    X = []
    y = []

    for rest_set_name in intraTrain[0]:
        X.append(load(rest_set_name))
        y.append(0)

    for math_set_name in intraTrain[1]:
        X.append(load(math_set_name))
        y.append(1)

    for motor_set_name in intraTrain[2]:
        X.append(load(motor_set_name))
        y.append(2)

    for memory_set_name in intraTrain[3]:
        X.append(load(memory_set_name))
        y.append(3)

    X = np.stack(X)  # shape: (num_samples, 248, 35624)
    y = np.array(y)

    # normalized_data = z_norm(rest1)

    return X, y
    
def z_norm(data):
    normalized_data = data.astype(float).copy() #copy of original array

    for i in range((data.shape[0])):
        row = data[i,:]
        mu = np.mean(row)
        sigma = np.std(row)
        normalized_data[i,:] = (row - mu) / sigma
    
    return normalized_data

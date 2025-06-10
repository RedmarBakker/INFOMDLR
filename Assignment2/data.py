# (a) Choose a suitable deep learning model for the involved classification tasks. Justify your choice.
# (b) Compare the accuracy of the 2 types of classification, i.e. intra-subject and cross subject data using your model. Explain your results.
# (c) Explain the choices of hyper-parameters of your model architecture and analyze their influence on the results (for both 2 types of classification). How they are selected?
# (d) If there is a significant difference in training and testing accuracies, what could be a possible reason? What are the alternative models or approaches you would select? Select one and implement to further improve your results. Justify your choice.

import os
import glob
import math
import h5py
import numpy as np
import mne.filter
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal
import mne
from matplotlib.animation import FuncAnimation
from sklearn.utils import shuffle


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


def z_norm(data):
    normalized_data = data.astype(float).copy()  # copy of original array

    for channel_id in range(data.shape[0]):
        channel_data = data[channel_id, :]
        mu = np.mean(channel_data)
        sigma = np.std(channel_data)
        normalized_data[channel_id, :] = (channel_data - mu) / sigma

    return normalized_data

def pre_process(data, frame_step_size = 1):
    n_channels, n_values = data.shape[0], data.shape[1]
    n_steps = math.ceil(n_values / frame_step_size)

    processed_data = []

    for channel_id in range(n_channels):
        channel_data = data[channel_id, :]
        averaged_steps = []

        for step_id in range(n_steps):
            start = step_id * frame_step_size
            end = min((step_id + 1) * frame_step_size, n_values)
            step_data = channel_data[start:end]
            averaged_steps.append(np.mean(step_data))

        processed_data.append(averaged_steps)

    return np.array(processed_data)

def downsample(sample, sfreq_new):
    sfreq_original = 2043

    if sample.ndim != 2:
        raise ValueError(
            f"Input shape must be 2D (n_channels, n_measurements)"
            f"got shape {sample.shape}"
        )

    if sfreq_new <= 0 or sfreq_original <= 0:
        raise ValueError("Sampling frequencies must be positive.")

    #Standard resample factors
    up_factor = 1.0
    down_factor = 1.0

    # Calculate the resampling factors
    if sfreq_new == sfreq_original:
        # No actual resampling needed, just pass through (or apply implicit filter)
        print(f"Warning: New sampling frequency ({sfreq_new} Hz) is the same as "
              f"original ({sfreq_original} Hz). No downsampling will occur, "
              f"but filtering might still be applied.")
    elif sfreq_new > sfreq_original:
        # This is upsampling, calculate up factor, down is 1.0
        up_factor = float(sfreq_new) / sfreq_original
        down_factor = 1.0
        print(f"Warning: New sampling frequency ({sfreq_new} Hz) is higher than "
              f"original ({sfreq_original} Hz). Upsampling will occur.")
    else:  # sfreq_new < sfreq_original, true downsampling
        up_factor = 1.0
        down_factor = float(sfreq_original) / sfreq_new

    downsampled_data = mne.filter.resample(sample,
                                           up=up_factor,
                                           down=down_factor,
                                           npad='auto',
                                           verbose=False)
    return downsampled_data



# step_size: the number of frames that will be averaged to make the data smaller
def build_dataset(sfreq_new, step_size=1):
    X = []
    y = []

    for rest_set_name in rest_set_names:
        X.append(z_norm(downsample(load(rest_set_name), sfreq_new=sfreq_new)))
        y.append(0)

    for math_set_name in math_set_names:
        X.append(z_norm(downsample(load(math_set_name), sfreq_new=sfreq_new)))
        y.append(1)

    for motor_set_name in motor_set_names:
        X.append(z_norm(downsample(load(motor_set_name), sfreq_new=sfreq_new)))
        y.append(2)

    for memory_set_name in memory_set_names:
        X.append(z_norm(downsample(load(memory_set_name), sfreq_new=sfreq_new)))
        y.append(3)

    X = np.stack(X)  # shape: (num_samples, 248, 35624)
    y = np.array(y)

    X, y = shuffle(X, y, random_state=42)

    return X, y

def create_cross_validation_sets(X, y, chunks=4):
    chunk_size = X.shape[0] // chunks
    cv_sets = []

    for i in range(chunks):
        for attempt in range(100):  # limit attempts to avoid infinite loop
            val_start = i * chunk_size
            val_end = (i + 1) * chunk_size

            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]

            if all(label in y_val for label in range(4)):
                X_train = np.concatenate((X[:val_start], X[val_end:]), axis=0)
                y_train = np.concatenate((y[:val_start], y[val_end:]), axis=0)
                cv_sets.append((X_train, y_train, X_val, y_val))
                break
            else:
                X, y = shuffle(X, y, random_state=i * 100 + attempt)
        else:
            raise ValueError("Unable to create a validation set with all classes present after 100 attempts.")

    return cv_sets
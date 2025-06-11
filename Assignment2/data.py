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
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, filepaths,
                 batch_size, new_sfreq,
                 shuffle=True, **kwargs
                 ):
        super().__init__(**kwargs)
        self.filepaths = filepaths
        self.batch_size = batch_size
        self.new_sfreq = new_sfreq
        self.shuffle = shuffle

        self.label_patterns = {
            "rest": 0,
            "task_story_math": 1,
            "task_motor": 2,
            "task_working_memory": 3
        }

        # This list will store references to the files (file index, label)
        self._all_sample_references = []

        # Enumerate over all filepaths and create references
        for filepath_idx, filepath in enumerate(self.filepaths):
            file_basename = os.path.basename(filepath)
            file_label = -1
            found_label = False
            for pattern, label_val in self.label_patterns.items():
                if file_basename.startswith(pattern):
                    file_label = label_val
                    found_label = True
                    break

            if not found_label:
                print(f"Warning: Could not determine label for file {filepath}. Skipping.")
                continue

            # We don't need to inspect total_measurements here, as we'll downsample the whole thing
            self._all_sample_references.append({
                'filepath_idx': filepath_idx,
                'label': file_label
            })

            self.on_epoch_end()  # Initialize indices and shuffle

    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch
        self.indices = np.arange(len(self._all_sample_references))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self._all_sample_references) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        X_batch_samples = []
        y_batch_labels = []

        for ref_idx in indices:
            ref = self._all_sample_references[ref_idx]
            filepath = self.filepaths[ref['filepath_idx']]

            # --- Load the raw full subject data ---
            raw_subject_data = load(filepath)  # Shape: (n_channels, original_n_measurements)

            # Ensure data is 2D (channels, time_steps) if load() returns (1, C, t)
            if raw_subject_data.ndim == 3 and raw_subject_data.shape[0] == 1:
                raw_subject_data = raw_subject_data.squeeze(axis=0)  # (n_channels, original_n_measurements)

            # --- Downsample the subject data ---
            downsampled_subject_data = downsample(
                raw_subject_data, self.new_sfreq
            )  # Shape: (n_channels, new_n_measurements)

            # --- Z-normalize the downsampled data ---
            # This is applied to the whole downsampled subject's recording (per channel)
            normalized_subject_data = z_norm(downsampled_subject_data)  # Shape: (n_channels, new_n_measurements)

            X_batch_samples.append(normalized_subject_data)
            y_batch_labels.append(ref['label'])

        X_batch = np.stack(X_batch_samples)  # Shape: (batch_size, n_channels, new_n_measurements, 1)
        y_batch = np.array(y_batch_labels)  # Shape: (batch_size,)

        return X_batch, y_batch


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

def load_files(directory_path, file_extension='.h5'):
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")

    all_filepaths = []
    # Use os.walk to find files in subdirectories too
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(file_extension):
                all_filepaths.append(os.path.join(root, file))

    # Sort file paths for consistent order before splitting (important for reproducibility without random_state)
    all_filepaths.sort()

    return all_filepaths

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



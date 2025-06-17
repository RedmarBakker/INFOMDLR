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
import keras
import matplotlib.pyplot as plt
from scipy import signal
import mne
from matplotlib.animation import FuncAnimation
from sklearn.utils import shuffle


# Build the cross train dataset
# Since the cross train dataset is too large to fit in memory, it has to be loaded in batches

# Data generator class for cross-subject training:
class AutoEncoderDataGenerator(keras.utils.Sequence):
    def __init__(self, filepaths, batch_size, chunk_size=None, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.filepaths = filepaths
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle

        self.indices = np.array([])

        if self.chunk_size is None:
            raise ValueError("For 'encoder' task_type, 'chunk_size' must be provided.")

        self.label_patterns = {
            "rest": 0, "task_story_math": 1, "task_motor": 2, "task_working_memory": 3
        }

        self._all_sample_references = []

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

            try:
                with h5py.File(filepath, 'r') as f:
                    data_name = list(f.keys())[0]
                    _, original_n_measurements_for_file = f[data_name].shape
            except Exception as e:
                print(f"Error reading metadata from {filepath}: {e}. Skipping file.")
                continue

            n_chunks_in_file = original_n_measurements_for_file // self.chunk_size

            for i in range(n_chunks_in_file):
                self._all_sample_references.append({
                    'filepath_idx': filepath_idx,
                    'start_time_idx': i * self.chunk_size,
                    'end_time_idx': (i + 1) * self.chunk_size,
                    'label': file_label  # <--- CORRECT: Store the actual label here
                })

        self.on_epoch_end()
    def on_epoch_end(self):
        self.indices = np.arange(len(self._all_sample_references))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self._all_sample_references) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        X_batch_samples = []
        y_batch_labels = []

        loaded_full_subject_data_cache = {}

        for ref_idx in indices:
            ref = self._all_sample_references[ref_idx]
            filepath = self.filepaths[ref['filepath_idx']]

            try:

                if filepath not in loaded_full_subject_data_cache:
                    raw_subject_data = load(filepath)
                    if raw_subject_data.ndim == 3 and raw_subject_data.shape[0] == 1:
                        raw_subject_data = raw_subject_data.squeeze(axis=0)

                    normalized_data = z_norm(raw_subject_data)

                    loaded_full_subject_data_cache[filepath] = normalized_data

                processed_full_subject_data = loaded_full_subject_data_cache[filepath]

                if 'start_time_idx' not in ref:
                    raise ValueError(f"Encoder reference missing chunk indices: {ref}")

                chunk = processed_full_subject_data[:, ref['start_time_idx']: ref['end_time_idx']]
                final_X_sample = chunk.reshape(chunk.shape[0], chunk.shape[1], 1)  # Correct: yields the actual label for the chunk

            except Exception as e:
                # --- CRITICAL DEBUG OUTPUT ---
                print(
                    f"ERROR in DataGenerator __getitem__: Failed to process file '{filepath}' (Index {ref_idx}, Batch {idx}). "
                    f"Specific error: {type(e).__name__}: {e}. Skipping this sample in batch.")
                continue

        if not X_batch_samples:
            print(
                f"WARNING: Batch {idx} is empty. All {len(indices)} requested samples for this batch failed to process or were skipped. "
                f"This will cause ValueError: need at least one array to stack.")

        X_batch = np.stack(X_batch_samples)
        y_batch = np.array(y_batch_labels)

        return X_batch, y_batch  # Autoencoder's pretext task is X as its own Y

class TransformerDataGenerator(keras.utils.Sequence):
    def __init__(self, filepaths, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.filepaths = filepaths
        self.shuffle = shuffle
        self.indices = np.array([])

        self.label_patterns = {
            "rest": 0, "task_story_math": 1, "task_motor": 2, "task_working_memory": 3
        }

        self._all_sample_references = []

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

            raw_subject_data = load(filepath)
            if raw_subject_data.ndim == 3 and raw_subject_data.shape[0] == 1:
                raw_subject_data = raw_subject_data.squeeze(axis=0)

            normalized_data = z_norm(downsample(raw_subject_data, 1022))

            self._all_sample_references.append(
                (extract_patches(normalized_data), file_label)
            )

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self._all_sample_references))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self._all_sample_references)

    def __getitem__(self, idx):
        patches, label = self._all_sample_references[self.indices[idx]]
        X = np.expand_dims(patches, axis=0)  # shape: (1, num_features)
        y = np.array([label])  # shape: (1,)
        return X, y

    def get_shape(self):
        return self._all_sample_references[self.indices[0]][0].shape

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
def build_dataset(filepaths, sfreq_new=1022):
    X = []
    y = []

    label_patterns = {
        "rest": 0, "task_story_math": 1, "task_motor": 2, "task_working_memory": 3
    }

    for filepath_idx, filepath in enumerate(filepaths):
        file_basename = os.path.basename(filepath)
        file_label = -1
        for pattern, label_val in label_patterns.items():
            if file_basename.startswith(pattern):
                file_label = label_val
                break

        if file_label == -1:
            raise ValueError(f"File {filepath} does not match any label.")

        X.append(extract_patches(z_norm(load(filepath))))
        y.append(file_label)

    X = np.stack(X)  # shape: (num_samples, 248, 35624)
    y = np.array(y)

    X, y = shuffle(X, y, random_state=42)

    return X, y

def create_cross_validation_sets(X, y, chunks=4, max_attempts=100):
    for attempt in range(max_attempts):
        X_shuffled, y_shuffled = shuffle(X, y, random_state=42 + attempt)
        chunk_size = X.shape[0] // chunks
        cv_sets = []
        all_folds_valid = True

        for i in range(chunks):
            val_start = i * chunk_size
            val_end = (i + 1) * chunk_size

            X_val = X_shuffled[val_start:val_end]
            y_val = y_shuffled[val_start:val_end]

            if all(label in y_val for label in [0, 1, 2, 3]):
                X_train = np.concatenate((X_shuffled[:val_start], X_shuffled[val_end:]), axis=0)
                y_train = np.concatenate((y_shuffled[:val_start], y_shuffled[val_end:]), axis=0)
                cv_sets.append((X_train, y_train, X_val, y_val))
            else:
                all_folds_valid = False
                break

        if all_folds_valid:
            return cv_sets

    raise ValueError(f"Could not create valid cross-validation sets with all labels after {max_attempts} shuffles.")

def glue_chunks(data:np.array, nr_chunks:int) -> np.array:
    if not nr_chunks:
        raise ValueError("No chunks provided to glue.")

    total_chunks, sensors, measurements, channels = data.shape
    num_groups = total_chunks // nr_chunks  # 1668 / 278 = 6


    subjects = []
    for i in range(num_groups):
        chunk_group = data[i * nr_chunks: (i + 1) * nr_chunks]# (278, 248, 64, 1)
        chunk_group = chunk_group.squeeze(-1)  # → (278, 248, 64)
        chunk_group = np.transpose(chunk_group, (1, 0, 2))  # → (248, 278, 64)
        split_chunks = [chunk_group[:, i, :] for i in range(nr_chunks)]  # list of 278 arrays of shape (248, 64)
        concatenated = np.concatenate(split_chunks, axis=1)      # → (248, 17792)
        subjects.append(concatenated)

    # Stack all groups into shape (subjects, 248, 17792, 1)
    result = np.stack(subjects, axis=0)
    return result

def extract_patches(data):
    """
    Args:
        data: (248, 35624, 1)
        patch_size: (248, 248)

    Returns:
        patches: (144, 248*248*1)
    """

    # data = np.squeeze(data, axis=-1)  # now (248, 35624)
    patch_size = data.shape[0]
    num_patches = math.floor((data.shape[0] * data.shape[1]) / (patch_size * patch_size))

    h, w = data.shape
    ph, pw = patch_size, patch_size
    assert h == ph

    usable_width = num_patches * pw
    data = data[:, :usable_width]

    patches = np.reshape(data, (num_patches, ph, pw))  # (144, 248, 248)
    patches = patches.reshape((num_patches, -1))       # (144, 61504)
    return patches

class CustomAccuracyLossCheckpoint(tf.keras.callbacks.Callback):
    """
    A custom ModelCheckpoint callback that saves the model based on validation accuracy
    first, and then uses validation loss as a tie-breaker.

    Saves if:
    1. val_accuracy is strictly greater than the best val_accuracy seen so far.
    2. OR, if val_accuracy is equal (or very close) to the best val_accuracy,
       AND val_loss is strictly lower than the best val_loss recorded at that best accuracy.
    """

    def __init__(self, filepath, verbose=0):
        super().__init__()
        self.filepath = filepath
        self.verbose = verbose

        # Initialize best metrics.
        # We want to maximize accuracy, so start with negative infinity.
        self.best_val_accuracy = -np.inf
        # We want to minimize loss at best accuracy, so start with positive infinity.
        self.best_val_loss_at_best_accuracy = np.inf

        # self.model will be set by Keras when the callback is added to model.fit()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val_accuracy = logs.get('val_accuracy')
        current_val_loss = logs.get('val_loss')

        # Check if metrics are available in logs
        if current_val_accuracy is None or current_val_loss is None:
            if self.verbose > 0:
                print(
                    f"Skipping CustomAccuracyLossCheckpoint for epoch {epoch + 1}: 'val_accuracy' or 'val_loss' not found in logs.")
            return

        model_saved = False  # Flag to indicate if model was saved in this epoch

        # --- Condition 1: Strictly better validation accuracy ---
        if current_val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = current_val_accuracy
            self.best_val_loss_at_best_accuracy = current_val_loss  # Update best loss for this new best accuracy

            self.model.save(self.filepath)
            model_saved = True
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: val_accuracy improved to {current_val_accuracy:.4f}, "
                      f"val_loss at this accuracy is {current_val_loss:.4f}. Saving model to {self.filepath}")

        # --- Condition 2: Equal validation accuracy (or very close due to float precision)
        #                 AND strictly better validation loss ---
        elif np.isclose(current_val_accuracy, self.best_val_accuracy, atol=1e-6):  # Use tolerance for float comparison
            if current_val_loss < self.best_val_loss_at_best_accuracy:
                self.best_val_loss_at_best_accuracy = current_val_loss  # Update best loss (accuracy remains same)

                self.model.save(self.filepath)
                model_saved = True
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: val_accuracy maintained at {current_val_accuracy:.4f}, "
                          f"val_loss improved to {current_val_loss:.4f}. Saving model to {self.filepath}")

        # Optional: Print message if no improvement
        if not model_saved and self.verbose > 1:  # verbose level 2 to show non-saving messages
            print(
                f"Epoch {epoch + 1}: val_accuracy ({current_val_accuracy:.4f}) and val_loss ({current_val_loss:.4f}) did not improve over best.")



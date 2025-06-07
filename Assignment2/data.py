# (a) Choose a suitable deep learning model for the involved classification tasks. Justify your choice.
# (b) Compare the accuracy of the 2 types of classification, i.e. intra-subject and cross subject data using your model. Explain your results.
# (c) Explain the choices of hyper-parameters of your model architecture and analyze their influence on the results (for both 2 types of classification). How they are selected?
# (d) If there is a significant difference in training and testing accuracies, what could be a possible reason? What are the alternative models or approaches you would select? Select one and implement to further improve your results. Justify your choice.
import math

import h5py
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib.animation import FuncAnimation
from sklearn.utils import shuffle


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

# step_size: the number of frames that will be averaged to make the data smaller
def build_dataset(step_size=1):
    X = []
    y = []

    for rest_set_name in rest_set_names:
        X.append(z_norm(pre_process(load(rest_set_name), step_size)))
        y.append(0)

    for math_set_name in math_set_names:
        X.append(z_norm(pre_process(load(math_set_name), step_size)))
        y.append(1)

    for motor_set_name in motor_set_names:
        X.append(z_norm(pre_process(load(motor_set_name), step_size)))
        y.append(2)

    for memory_set_name in memory_set_names:
        X.append(z_norm(pre_process(load(memory_set_name), step_size)))
        y.append(3)

    X = np.stack(X)  # shape: (num_samples, 248, 32520)
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
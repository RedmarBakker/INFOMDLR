# (a) Choose a suitable deep learning model for the involved classification tasks. Justify your choice.
# (b) Compare the accuracy of the 2 types of classification, i.e. intra-subject and cross subject data using your model. Explain your results.
# (c) Explain the choices of hyper-parameters of your model architecture and analyze their influence on the results (for both 2 types of classification). How they are selected?
# (d) If there is a significant difference in training and testing accuracies, what could be a possible reason? What are the alternative models or approaches you would select? Select one and implement to further improve your results. Justify your choice.


import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mne
from matplotlib.animation import FuncAnimation

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

def build_dataset():
    X = []
    y = []

    for rest_set_name in rest_set_names:
        temp = signal.decimate(load(rest_set_name),q=10)
        X.append(z_norm(temp))
        y.append(0)

    for math_set_name in math_set_names:
        temp = signal.decimate(load(math_set_name),q=10)
        X.append(z_norm(temp))
        y.append(1)

    for motor_set_name in motor_set_names:
        temp = signal.decimate(load(motor_set_name),q=10)
        X.append(z_norm(temp))
        y.append(2)

    for memory_set_name in memory_set_names:
        temp = signal.decimate(load(memory_set_name),q=10)
        X.append(z_norm(temp))
        y.append(3)

    X = np.stack(X)  # shape: (num_samples, 248, 32520)
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

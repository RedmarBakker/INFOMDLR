import numpy as np

def create_training_data(data, history_length):
    x = []
    y = []

    for i in range(len(data) - history_length):
        x.append(data[i:i + history_length])
        y.append(data[i + history_length])

    return np.array(x), np.array(y)

def scale_data(data, min_val=1, max_val=255):
    return (data - min_val) / (max_val - min_val)
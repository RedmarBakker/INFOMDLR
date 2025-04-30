import numpy as np

def create_training_data(data, history_length):
    x = []
    y = []

    for i in range(len(data) - history_length):
        x.append(data[i:i + history_length])
        y.append(data[i + history_length])

    return np.array(x), np.array(y)
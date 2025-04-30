import numpy as np

def create_training_data(data, sequence_length):
    x = []
    y = []

    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])

    return np.array(x), np.array(y)
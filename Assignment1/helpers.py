import numpy as np

def create_training_data(data, history_length):
    x, y = [], []
    for i in range(len(data) - history_length):
        x.append(data[i:i+history_length,0])
        y.append(data[i+history_length,0])
    return np.array(x)[:, np.newaxis], np.array(y)[:, np.newaxis]

def scale_data(data, min_val=1, max_val=255):
    return (data - min_val) / (max_val - min_val)

def scale_x_axis(epochs) -> np.array:
    tick_locations = np.array([1])
    multiples_of_10 = np.arange(10, epochs + 1, 10)
    tick_locations = np.concatenate((tick_locations, multiples_of_10))
    if epochs > 1 and epochs not in tick_locations:
        tick_locations = np.append(tick_locations, epochs)
    return tick_locations

def generate_sequence(x_val, model, history_length, num_steps):
    """Generates a sequence of predicted data points using the trained model."""
    """Returns a list of the predicted data points."""
    # Start with the last history_length data points from x_val
    sequence = x_val[-history_length:].reshape(history_length, 1)
    predicted_data_points = []

    # Predict the next data point, add it to the sequence and
    # remove the first data point
    for _ in range(num_steps):
        input_seq = sequence[np.newaxis, ...]
        predicted_output = model.predict(input_seq)
        sequence = np.append(sequence[1:], [predicted_output[0]], axis=0)

        # Scale the predicted output back to the original range
        predicted_output_scaled = predicted_output[0] * 255
        predicted_data_points.append(predicted_output_scaled)


    predicted_data_points = np.array(predicted_data_points)
    predicted_data_points = predicted_data_points.astype(np.uint8)
    return predicted_data_points

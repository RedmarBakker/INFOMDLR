import keras
import numpy as np

def build_autoencoder(filters=32, chunk_size=64, compression_factor=2):
    # Input shape (e.g., for 28x28 grayscale images like MNIST)
    input_shape = (1, 248, chunk_size)

    # Encoder
    input_data = keras.layers.Input(shape=input_shape)

    x = keras.layers.Conv2D(filters, (1, 1), strides=(1,1), activation='relu', padding='same', data_format='channels_first')(input_data)

    # Residual block definition
    def residual_block(input_tensor):
        residual = keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=(1,1), padding='same', activation='relu', data_format='channels_first')(input_tensor)
        return keras.layers.Add()([input_tensor, residual])

    # Apply 4 residual blocks
    for _ in range(4):
        x = residual_block(x)

    # Final convolution to reduce t -> t/2
    # We'll use strides=(2,1) to halve only the height (t dimension)
    x = keras.layers.Conv2D(1, kernel_size=(1, 1), strides=(1, compression_factor), padding='same', activation='relu', data_format='channels_first')(x)

    # This is your latent representation
    encoder_output = x

    # Create encoder model
    encoder_model = keras.Model(inputs=input_data, outputs=encoder_output)

    #encoder_model.summary()

    return encoder_model

def chunk_data(X, chunk_size=64):
    # (1, 248, 2043 * 17.5)

    _, height, width = X.shape

    n_chunks = width // chunk_size
    chunks = np.stack([X[:, :, i * chunk_size: (i + 1) * chunk_size] for i in range(n_chunks)])

    return chunks

def glue_chunks(chunks):
    return np.concatenate(chunks, axis=-1)
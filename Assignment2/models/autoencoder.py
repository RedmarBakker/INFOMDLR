import keras
import numpy as np


def build_autoencoder(filters, sensors, chunk_size,
                      compression_factor,
                      dropout=0.0, l2_factor = 0.0,
                      data_format='channels_last'):
    # Build encoder
    encoder = build_encoder(filters=filters,
                            chunk_size=chunk_size,
                            sensors=sensors,
                            compression_factor=compression_factor,
                            dropout=dropout,
                            data_format=data_format,
                            l2_factor=l2_factor)

    # Latent shape: encoder.output_shape will be (None, 248, 64, 1) for channels_last
    latent_shape = encoder.output_shape[1:] # Slices to get (248, 64, 1)

    # Build decoder
    decoder = build_decoder(latent_shape=latent_shape,
                            filters=filters,
                            compression_factor=compression_factor,
                            l2_factor=l2_factor)

    # Create autoencoder model
    # Autoencoder's input shape must match encoder's expected input shape
    autoencoder_inputs = keras.layers.Input(shape=(sensors, chunk_size, 1)) # Changed this shape to channels_last

    latent_representation = encoder(autoencoder_inputs)
    autoencoder_outputs = decoder(latent_representation)

    autoencoder_model = keras.Model(inputs=autoencoder_inputs,
                                    outputs=autoencoder_outputs,
                                    name='autoencoder')

    return autoencoder_model, encoder, decoder

def build_encoder(filters=32, chunk_size=128,
                  sensors=248, compression_factor=2,
                  dropout = 0.0, l2_factor = 0.0,
                  data_format='channels_last'):

    input_shape = (sensors, chunk_size, 1)
    input_data = keras.layers.Input(shape=input_shape)

    regularizer = keras.regularizers.l2(l2_factor) if l2_factor > 0.0 else None

    # First convolution layer
    x = keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=(1,1),
                            padding='same', kernel_regularizer=regularizer,
                            data_format=data_format)(input_data)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    if dropout > 0.0:
        x = keras.layers.Dropout(dropout)(x)

    # Apply 4 residual blocks
    for _ in range(4):
        x = residual_block(filters=filters, input_tensor=x,
                           dropout=dropout, l2_factor=l2_factor,
                           data_format=data_format)

    # Final convolution to reduce t -> t/2 (width dimension)
    encoder_output = keras.layers.Conv2D(1, kernel_size=(1, 1), # filters=1 because final output has 1 channel
                            strides=(1, compression_factor), padding='same',
                            kernel_regularizer=regularizer, activation='relu',
                            data_format=data_format)(x)

    encoder_model = keras.Model(inputs=input_data,
                                outputs=encoder_output,
                                name='encoder')
    return encoder_model

def build_decoder(latent_shape, filters=32, compression_factor=2, l2_factor = 0.0):
    # latent_shape will be (248, 32, 1) for channels_last
    latent_input = keras.layers.Input(shape=latent_shape)

    regularizer = keras.regularizers.l2(l2_factor) if l2_factor > 0.0 else None

    # Decoder's first layer: Standard Conv2D (channels_last)
    x = keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=(1, 1),
                        padding='same', kernel_regularizer=regularizer,
                        activation='relu',
                        data_format='channels_last')(latent_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Apply 4 residual blocks
    for _ in range(4):
        x = residual_block(filters=filters, input_tensor=x, l2_factor=l2_factor)

    # Decoder's last layer: Conv2DTranspose (channels_last)
    output = keras.layers.Conv2DTranspose(filters=1,
                                          kernel_size=(1, compression_factor),
                                          strides=(1, compression_factor),
                                          padding='same',
                                          kernel_regularizer=regularizer,
                                          data_format='channels_last',
                                          activation="linear")(x) # Use sigmoid/tanh for scaled data

    decoder_model = keras.Model(inputs=latent_input, outputs=output, name='decoder')
    return decoder_model


# Residual block definition
def residual_block(filters, input_tensor, data_format='channels_last',
                   dropout = 0.0, l2_factor = 0.0):

    regularizer = keras.regularizers.l2(l2_factor) if l2_factor > 0.0 else None

    residual = keras.layers.Conv2D(filters, kernel_size=(1, 3), strides=(1,1),
                                   padding='same', kernel_regularizer=regularizer,
                                   data_format=data_format)(input_tensor)
    residual = keras.layers.BatchNormalization()(residual)

    output = keras.layers.Add()([input_tensor, residual])
    output = keras.layers.Activation('relu')(output)

    if dropout > 0.0:
        output = keras.layers.Dropout(dropout)(output)

    return output

def chunk_data(X, chunk_size=64):
    # (1, 248, 2043 * 17.5)

    _, height, width = X.shape

    n_chunks = width // chunk_size
    chunks = np.stack([X[:, :, i * chunk_size: (i + 1) * chunk_size] for i in range(n_chunks)])

    return chunks

def chunk_data_for_conv2d(X, chunk_size):
    # X shape: (num_samples, channels, total_measurements) e.g., (32, 248, 35624)
    num_samples, num_channels, total_measurements = X.shape

    n_chunks_per_sample = total_measurements // chunk_size

    # Reshape each sample into chunks:
    # From (num_samples, num_channels, total_measurements)
    # To (num_samples, num_channels, n_chunks_per_sample, chunk_size)
    reshaped_X = X[:, :, :n_chunks_per_sample * chunk_size].reshape(
        num_samples, num_channels, n_chunks_per_sample, chunk_size
    )

    # Transpose and reshape for channels_last format:
    # Desired: (num_total_chunks, Height, Width, Channels)
    # Here, Height = num_channels, Width = chunk_size, Channels = 1 (dummy for Conv2D)
    # So, transpose from (num_samples, num_channels, n_chunks_per_sample, chunk_size)
    # To (num_samples, n_chunks_per_sample, num_channels, chunk_size)
    # Then reshape to (num_total_chunks, num_channels, chunk_size, 1)
    chunks = reshaped_X.transpose(0, 2, 1, 3).reshape(
        num_samples * n_chunks_per_sample, num_channels, chunk_size, 1 # Added the final '1' for channels_last
    )
    return chunks

def glue_chunks(chunks):
    return np.concatenate(chunks, axis=-1)



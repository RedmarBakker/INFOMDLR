import keras
import numpy as np
#from models.autoencoder import chunk_data 

def temp_block(input, filters= 16, dropout = 0.0, l2fac = 0.0):

    #input = keras.layers.Input(shape=input_shape)
    reguliser = keras.regularizers.L2(l2fac) if l2fac > 0.0 else None
    x = keras.layers.Conv2D(filters, kernel_size=(1, 2), strides=(1,2),
                            padding='same', data_format='channels_last',
                            kernel_regularizer=reguliser)(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    if dropout > 0.0:
        x = keras.layers.Dropout(dropout)(x)

    # Residual block definition
    def residual_block(input_tensor, dropout = 0.0, reguliser = None):
        residual = keras.layers.Conv2D(filters, kernel_size=(1, 3), strides=(1,1),
                                       padding="same", data_format='channels_last',
                                       kernel_regularizer=reguliser)(input_tensor)
        norm = keras.layers.BatchNormalization()(residual)
        output = keras.layers.Add()([input_tensor, norm])
        output = keras.layers.Activation('relu')(output)

        if dropout > 0:
            output = keras.layers.Dropout(dropout)(output)

        return output

    # Apply 4 residual blocks
    for _ in range(4):
        x = residual_block(x, dropout=dropout, reguliser=reguliser)

    return x

def build_spat_temp_model(input_shape, embed_dim = 16, dropout = 0.0, l2fac = 0.0):

    num_sens = input_shape[0]
    num_t = input_shape[1]

    input = keras.layers.Input(shape=input_shape) #1 is for the channels
    reguliser = keras.regularizers.L2(l2fac) if l2fac > 0.0 else None
    
    reshaped_input = keras.layers.Reshape((num_sens, num_t, 1))(input)

    x = keras.layers.Conv2D(filters=embed_dim, kernel_size= (248,1),
                            data_format='channels_last',
                            kernel_regularizer=reguliser)(reshaped_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    if dropout > 0.0:
        x = keras.layers.Dropout(dropout)(x)

    for _ in range(3):
        x = temp_block(x, filters=embed_dim, dropout=dropout)
    
    # Last convolution (1x1) and Flatten
    x_reduced_channels = keras.layers.Conv2D(filters=1, kernel_size=(1,1), data_format='channels_last', kernel_regularizer=reguliser)(x)
    x = keras.layers.BatchNormalization()(x_reduced_channels)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Flatten(name='flatten_features', data_format='channels_last')(x)

    x = keras.layers.Dense(64, name='fc1', activation='relu')(x)

    output_classification = keras.layers.Dense(4, activation='softmax', name='output_softmax')(x)

    model = keras.Model(inputs=input,outputs=output_classification)
    
    optim = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optim,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model


def spat_lstm(input_shape = (248, 7125), embed_dim = 8):
    num_sens = 248
    num_t = 7125 #3563 if downsampling by factor of 10
    input = keras.layers.Input(shape=input_shape) #1 is for the channels
    
    reshaped_input = keras.layers.Reshape((1, num_sens, num_t))(input)
    print(reshaped_input.shape)

    x = keras.layers.Conv2D(filters=embed_dim, kernel_size= (248,1),data_format='channels_last')(reshaped_input) #new shape:(None,embed_dim,1,timesteps)

    x = keras.layers.Reshape(target_shape= (embed_dim, num_t))(x) #get rid of 1
    #x = keras.layers.Reshape(target_shape= (num_t,embed_dim)) #shape for LSTM
    permuted_for_lstm = keras.layers.Permute((2, 1), name='permute_for_lstm')(x)
    #print(f"Shape after spatial embedding but before LSTM{x.shape}")
    permuted_for_lstm = keras.layers.BatchNormalization()(permuted_for_lstm)
    lstm_layer = keras.layers.LSTM(16,dropout=0.3,recurrent_dropout=0.3)(permuted_for_lstm)
    #print(f"Shape after lstm: {lstm_layer}")

    output_classification = keras.layers.Dense(4, activation='softmax', name='output_softmax')(lstm_layer)

    # --- Create Model ---
    optim = keras.optimizers.Adam(learning_rate=0.0005)
    model = keras.Model(inputs=input, outputs=output_classification, name='Spatial_LSTM_Model')
    model.compile(optimizer=optim,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    model.summary()
    return model


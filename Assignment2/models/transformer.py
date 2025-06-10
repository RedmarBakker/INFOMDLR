import keras
import math

def build_transformer(data_length=math.floor(2043*17.5), embedding_dim=16, n_layers=4, n_attn_heads=4, attn_dropout=0.1, ffn_dropout=0.1, final_dense_units=64, final_dropout=0.1):
    # Input shape (e.g., for 28x28 grayscale images like MNIST)
    input_shape = (248, data_length)

    # Encoder
    input_data = keras.layers.Input(shape=input_shape)

    x = keras.layers.Dense(embedding_dim, activation='relu')(input_data)

    # Transformer Encoder block
    for _ in range(n_layers):
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = keras.layers.MultiHeadAttention(num_heads=n_attn_heads, key_dim=embedding_dim)(x, x)
        attn_output = keras.layers.Dropout(attn_dropout)(attn_output)
        x = keras.layers.Add()([x, attn_output])
        ffn_output = keras.layers.Dense(embedding_dim * 4, activation='relu')(x)
        ffn_output = keras.layers.Dropout(ffn_dropout)(ffn_output)
        ffn_output = keras.layers.Dense(embedding_dim)(ffn_output)
        x = keras.layers.Add()([x, ffn_output])

    # Global average pooling and classification head
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(final_dense_units, activation='relu')(x)
    x = keras.layers.Dropout(final_dropout)(x)

    outputs = keras.layers.Dense(4, activation='softmax')(x)

    model = keras.models.Model(inputs=input_data, outputs=outputs)

    # Dynamically calculate learning rate based on model complexity
    base_lr = 0.00004
    complexity_factor = (1.25 * embedding_dim * n_layers * final_dense_units) / 1024.0
    learning_rate = base_lr / complexity_factor

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
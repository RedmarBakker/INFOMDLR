import keras
import math

def build_transformer(data_length=math.floor(2043*17.5), embedding_dim=16, layers=4, num_classes=4):
    # Input shape (e.g., for 28x28 grayscale images like MNIST)
    input_shape = (248, data_length)

    # Encoder
    input_data = keras.layers.Input(shape=input_shape)

    x = keras.layers.Dense(embedding_dim, activation='relu')(input_data)

    # Transformer Encoder block
    for _ in range(layers):
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = keras.layers.MultiHeadAttention(num_heads=4, key_dim=embedding_dim)(x, x)
        x = keras.layers.Add()([x, attn_output])
        ffn_output = keras.layers.Dense(embedding_dim * 4, activation='relu')(x)
        ffn_output = keras.layers.Dense(embedding_dim)(ffn_output)
        x = keras.layers.Add()([x, ffn_output])

    # Global average pooling and classification head
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    # x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.models.Model(inputs=input_data, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
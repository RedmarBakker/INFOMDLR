import tensorflow as tf
import keras
import math

class ClassToken(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]
        cls = tf.broadcast_to(self.w, shape=(batch_size, 1, hidden_dim))
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls


def mlp(x, config:dict):
    x = keras.layers.Dense(config['mlp_dim'], activation='gelu')(x)
    x = keras.layers.Dropout(config['dropout_rate'])(x)
    x = keras.layers.Dense(config['embedding_size'])(x)
    x = keras.layers.Dropout(config['dropout_rate'])(x)
    return x

def transformer_encoder(x, config:dict):
    res_1 = x
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = keras.layers.MultiHeadAttention(
        num_heads=config['num_heads'],
        key_dim=config['embedding_size'],
        dropout=config['dropout_rate']
    )(x, x)
    x = keras.layers.Add()([x, res_1])

    res_2 = x
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = mlp(x, config)
    x = keras.layers.Add()([x, res_2])

    return x

def build_transformer(cf:dict):
    """
    Builds a Transformer model based on the provided configuration.

    Args:
        cf (dict): Configuration dictionary containing model parameters.

        Returns:
            keras.Model: A Keras model instance representing the Transformer.
        """

    """Inputs"""
    inputs = (cf['num_patches'], cf['patch_size'] * cf['patch_size'] * cf['num_channels'])
    input_layer = keras.layers.Input(shape=inputs) #(None, 144, 61504)

    """Embeddings (Patch + Positional)"""
    patch_embeddings = keras.layers.Dense(cf['embedding_size'])(input_layer)  # (None, 144, 16)
    positions = tf.range(start=0, limit=cf['num_patches'], delta=1, dtype=tf.int32) # (144,)
    pos_embeddings = keras.layers.Embedding(input_dim=cf['num_patches'], output_dim=cf['embedding_size'])(positions)  # (144, 16)
    embeddings = patch_embeddings + pos_embeddings  # (None, 144, 16)

    """Class Token"""
    token = ClassToken()(embeddings)
    x = keras.layers.Concatenate(axis=1)([token, embeddings])  # (None, 145, 16)

    """Transformer Encoder"""
    for _ in range(cf['num_layers']):
        x = transformer_encoder(x, cf)

    """Classification Head"""
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x) # (None, 145, 16)
    x = x[:, 0, :]  # (None, 16) Selects the class token (first token) for classification
    x = keras.layers.Dropout(cf['dropout_rate'])(x)
    output_layer = keras.layers.Dense(4, activation='softmax')(x)  # 4 classes for classification

    model = keras.models.Model(inputs=input_layer, outputs=output_layer, name='Transformer')

    return model

if __name__ == '__main__':

    MegDataShape = (248, 35624)
    config = {}
    config['num_layers'] = 4
    config['embedding_size'] = 16
    config['num_heads'] = 4
    config['dropout_rate'] = 0.1
    config['num_channels'] = 1
    config['mlp_dim'] = 64
    config['patch_size'] = MegDataShape[0]
    config['num_patches'] = math.ceil((config['patch_size'] * MegDataShape[1]) / (config['patch_size'] * config['patch_size']))


    model = build_transformer(config)
    model.summary()
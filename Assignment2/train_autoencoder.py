import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib
# Use the TkAgg backend for matplotlib to avoid the need for a display
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from data import CrossSubjectDataGenerator
from models.autoencoder import build_autoencoder

# ---- TRAINING THE AUTOENCODER----
# hyperparameters
MODEL_CHUNK_SIZE = 128
BATCH_SIZE = 32
FILTERS = 32
COMPRESSION_FACTOR = 2
EPOCHS = 100
DROPOUT = 0.3
L2_FACTOR = 0.001
print("Start!")

# Select training setup
training_setup = ['intra', 'cross']

# Train both intra and cross subject models
for setup in training_setup:
    train_data_dir = []

    match training_setup:
        case 'intra':
            train_data_dir.append('./Intra/train')
        case 'cross':
            train_data_dir.append('./Cross/train')
        case 'both':
            train_data_dir.extend(['./Cross/train', './Intra/train'])
        case _:
            raise ValueError(f"Unknown training setup: {training_setup}")

    all_filepaths = []
    for directory in train_data_dir:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.h5'):
                    all_filepaths.append(os.path.join(root, file))
        all_filepaths.sort()

    if not all_filepaths:
        raise ValueError(f"No .h5 files found in {dir}. Check path and file extensions.")

    train_filepaths, val_filepaths = train_test_split(
        all_filepaths,
        test_size= 0.21 if training_setup == 'both' else 0.18, # 0.21 for intra + cross
        random_state=42)
    print("Split!")

    # Create train and validation generator
    train_generator = CrossSubjectDataGenerator(
            filepaths=train_filepaths,
            chunk_size=MODEL_CHUNK_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True)
    val_generator = CrossSubjectDataGenerator(
            filepaths=val_filepaths,
            chunk_size=MODEL_CHUNK_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False)

    autoencoder, encoder, decoder = build_autoencoder(filters=FILTERS,
                                                      sensors=248,
                                                      chunk_size=MODEL_CHUNK_SIZE,
                                                      compression_factor=COMPRESSION_FACTOR,
                                                      dropout=DROPOUT,
                                                      l2_factor=L2_FACTOR)

    autoencoder.compile(optimizer='adam', loss='mse')
    print("Compiled!")

    # --- Callbacks ---
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'models/autoencoder_{training_setup}_dropout{DROPOUT}+L2{L2_FACTOR}_tanh.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    print(f"\nStarting training using generator. Batches per training epoch: {len(train_generator)}")
    print(f"Batches per validation epoch: {len(val_generator)}")
    print(f"Model expected input shape: {autoencoder.input_shape}")

    # ---- TRAINING ----
    history = autoencoder.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
    print("\nAutoencoder training complete!")

    # --- Plot training history ---
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Autoencoder Training History ({training_setup}-Subject Data)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
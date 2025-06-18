import os
import tensorflow as tf
import keras
import numpy as np
from tensorflow.python.keras.utils.version_utils import training

from data import glue_chunks
from data import AutoEncoderDataGenerator, load_files
from models.transformer import ClassToken
from sklearn.model_selection import train_test_split
from data import build_dataset
import math

model_folder = './models/cnn/intra'

filepath = './Intra/train'
dataset_source = 'intra'

all_files = []
for root, _, files in os.walk(filepath):
    for file in files:
        if file.endswith('.h5'):
            all_files.append(os.path.join(root, file))

all_files.sort()
print(f"Found {len(all_files)} files in {filepath}")

with_autoencoder = False

if not with_autoencoder:
    X, y = build_dataset(all_files, with_patches=False, resampling=250)
else:
    # Select the autoencoder model
    match dataset_source:
        case 'intra':
            model_path = './models/autoencoder_intra_dropout+L2.keras'
        case 'cross':
            model_path = './models/autoencoder_cross_dropout+L2.keras'
        case _:
            raise ValueError(f"Unknown model: {dataset_source}")
    try:
        autoencoder = keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading the model from {model_path}: {e}")

    # Train-test split
    encoder = autoencoder.get_layer('encoder')
    print(f"Encoder loaded")

    # Hyperparameter
    # Encoder
    BATCH_SIZE_ENCODER = 32
    CHUNK_SIZE = 128

    EncoderTrainData = AutoEncoderDataGenerator(
        filepaths=load_files(filepath),
        batch_size=BATCH_SIZE_ENCODER,
        chunk_size=CHUNK_SIZE,
        shuffle=False
    )

    all_X_train_batches = []
    all_y_train_batches = []

    for i in range(len(EncoderTrainData)):  # Iterate through all batches
        X_batch, y_batch = EncoderTrainData[i]  # Get X and Y for each batch
        if X_batch.shape[0] > 0:  # Only append if the batch is not empty
            all_X_train_batches.append(X_batch)
            all_y_train_batches.append(y_batch)
        else:
            print(f"WARNING: Skipping empty batch {i} from encoder_train_generator during collection.")

    X_train_full_dataset = np.concatenate(all_X_train_batches, axis=0)
    y_train_full_dataset = np.concatenate(all_y_train_batches, axis=0)

    # Concatenate the labels
    nr_chunks = int(math.floor(35624 / CHUNK_SIZE))

    y = y_train_full_dataset[::nr_chunks]

    print(f"train data shape: {X_train_full_dataset.shape}")
    print(f"train labels shape: {y.shape}")

    # Create the latent representations and glue them for training
    X_train_latent_chunks = encoder.predict(X_train_full_dataset)

    X = glue_chunks(X_train_latent_chunks, nr_chunks=nr_chunks)

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Created train and validation sets with {len(X)} and {len(y)} samples respectively.")

evaluated_models_results = {}

for model_name in os.listdir(model_folder):
    if model_name.endswith('.keras'):
        model_path = os.path.join(model_folder, model_name)

        if not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist. Skipping.")
            continue

        print(f"Evaluating model: {model_name}")
        try:
            model = keras.models.load_model(model_path, {'ClassToken': ClassToken})
            loss, accuracy = model.evaluate(X, y, verbose=1)

            print(f"  Validation Loss: {loss:.6f}")
            print(f"  Validation Accuracy: {accuracy:.4f}")

            evaluated_models_results[model_name] = {'loss': loss, 'accuracy': accuracy}

        except Exception as e:
            print(f"  Error loading or evaluating {model_name}. Skipping. Specific error: {e}")
            continue

print("\n--- Evaluation Summary ---")
if evaluated_models_results:
    # Find the best model based on validation loss
    best_loss_model_name = None
    lowest_val_loss = float('inf')

    # Find the best model based on validation accuracy
    best_acc_model_name = None
    highest_val_accuracy = -float('inf')

    for model_name, metrics in evaluated_models_results.items():
        print(f"Model: {model_name} | Val Loss: {metrics['loss']:.6f} | Val Accuracy: {metrics['accuracy']:.4f}")

        if metrics['loss'] < lowest_val_loss:
            lowest_val_loss = metrics['loss']
            best_loss_model_name = model_name

        if metrics['accuracy'] > highest_val_accuracy:
            highest_val_accuracy = metrics['accuracy']
            best_acc_model_name = model_name

    print("\n--- Best Model(s) ---")
    print(f"Model with LOWEST Validation Loss: {best_loss_model_name} (Loss: {lowest_val_loss:.6f})")
    print(f"Model with HIGHEST Validation Accuracy: {best_acc_model_name} (Accuracy: {highest_val_accuracy:.4f})")
else:
    print("No models were successfully evaluated.")





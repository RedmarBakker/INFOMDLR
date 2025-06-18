from textwrap import indent

import keras

import platform

import tensorflow as tf

import math

from data import pre_process

tf.config.set_visible_devices([], 'GPU')

from keras.src.saving import load_model
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.python.distribute.multi_process_lib import multiprocessing

from models.transformer import ClassToken
from data import build_dataset, AutoEncoderDataGenerator, load_files, glue_chunks, create_cross_validation_sets
from models.transformer import build_transformer
import matplotlib.pyplot as plt
from itertools import product
import json
import os
from tabulate import tabulate
import numpy as np
from multiprocessing import Pool, Manager
from functools import partial

from keras.src.callbacks import Callback

class OverfitStopping(Callback):
    def __init__(self, factor=5.0, patience=5):
        super().__init__()
        self.factor = factor
        self.patience = patience
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        if train_loss is None or val_loss is None:
            return

        if train_loss * self.factor < val_loss:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nStopping early: train_loss is {self.factor}x smaller than val_loss for {self.patience} epochs.")
                self.model.stop_training = True
        else:
            self.wait = 0

# for i, (emb_dim, num_layers, n_heads, mlp_dims, dropout_rate) in enumerate(combinations):
def train_model(model_name, dataset_source, cv_sets, config, results_path):
    # Build the Transformer model

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for i, (X_train, y_train, X_val, y_val) in enumerate(cv_sets):

        # Now use these values to build your model
        transformer = build_transformer(config)

        transformer.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callback filepath
        filepath = f"./models/{model_name}/{dataset_source}/emb{config['embedding_size']}_layers{config['num_layers']}_heads{config['num_heads']}_mlp{config['mlp_dim']}_dropout{config['dropout_rate']}_cv{i}.keras"

        # # Callbacks
        # early_stopping = keras.callbacks.EarlyStopping(
        #     monitor='val_loss',
        #     patience=80,
        #     restore_best_weights=True
        # )

        overfit_stopping = OverfitStopping(factor=4, patience=10)
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            #verbose=1
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.98,
            patience=1,
            min_lr=1e-6,
            #verbose=1
        )

        transformer.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val), batch_size=16, callbacks=[model_checkpoint, reduce_lr, overfit_stopping], verbose=1)

        best_model = load_model(filepath, {'ClassToken': ClassToken})
        train_loss, train_accuracy = best_model.evaluate(X_train, y_train)
        val_loss, val_accuracy = best_model.evaluate(X_val, y_val)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_accuracy)

    config_result = {
        "embedding_dim": config['embedding_size'],
        "n_layers": config['num_layers'],
        "n_attn_heads": config['num_heads'],
        "mlp_dims": config['mlp_dim'],
        "dropout_rate": config['dropout_rate'],

        # "val_loss": val_loss,
        # "val_accuracy": val_accuracy,
        # "overfit_factor": val_loss / train_loss,
        "val_loss": np.mean(val_loss_list),
        "val_accuracy": np.mean(val_acc_list),
        "overfit_factor": np.mean(val_loss_list) / np.mean(train_loss_list),
    }

    with open(results_path, "r+") as f:
        try:
            all_results = json.load(f)
        except json.JSONDecodeError:
            all_results = []

        all_results.append(config_result)
        f.seek(0)

        json.dump(all_results, f, indent=2)

    return np.mean(val_acc_list), np.mean(val_loss_list)

def tune_transformer_parameters(embedding_dims, n_layers, n_attn_heads, mlp_dims, dropout_rate, reversed_execution=False, with_autoencoder=False):
    # Process data
    dataset_source = 'cross'  # intra or 'cross'

    print(f'Starting tuning process with source {dataset_source}...')

    match dataset_source:
        case 'intra':
            filepath = "./Intra/train"
        case 'cross':
            filepath = "./Cross/train"
        case _:
            raise ValueError(f"Unknown data type: {dataset_source}")

    # Collect all training files
    all_files = []
    for root, _, files in os.walk(filepath):
        for file in files:
            if file.endswith('.h5'):
                all_files.append(os.path.join(root, file))

    all_files.sort()

    if not all_files:
        raise ValueError(f"No .h5 files found in {dir}. Check path and file extensions.")

    n_folds = 5
    print(f'Building datasets with {n_folds} folds...')

    if with_autoencoder:
        latent_path = f'data/autoencoder/{dataset_source}/X_latent.npy'
        labels_path = f'data/autoencoder/{dataset_source}/y_labels.npy'

        if os.path.exists(latent_path) and os.path.exists(labels_path):
            print("Loading latent dataset from disk...")
            X = np.load(latent_path)
            y = np.load(labels_path)
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
            X = pre_process(X)

            os.makedirs(f'models/transformer_autoencoder/{dataset_source}', exist_ok=True)

            np.save(latent_path, X)
            np.save(labels_path, y)

            print(f"Saved latent data and labels to disk.")
    else:
        X, y = build_dataset(all_files)

    cv_sets = create_cross_validation_sets(X, y, chunks=n_folds)

    model_name = 'transformer'

    if with_autoencoder:
        model_name += '_autoencoder'

    results_path = f"models/{model_name}/{dataset_source}/results.json"

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Ensure the results file exists
    if not os.path.exists(results_path):
        with open(results_path, 'w') as f:
            json.dump([], f)

    print(f'Storing results in: {results_path}')

    configurations = list(product(embedding_dims, n_layers, n_attn_heads, mlp_dims, dropout_rate))

    if reversed_execution:
        configurations = reversed(configurations)

    for combination in configurations:
        (emb_dim, num_layers, n_heads, mlp_dims, dropout_rate) = combination

        # Load existing results
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                try:
                    all_results = json.load(f)
                except json.JSONDecodeError:
                    all_results = []
        else:
            all_results = []

        # Check if config already exists in results
        skip = False
        for result in all_results:
            if (result.get("embedding_dim") == emb_dim and
                    result.get("n_layers") == num_layers and
                    result.get("n_attn_heads") == n_heads and
                    result.get("mlp_dims") == mlp_dims and
                    result.get("dropout_rate") == dropout_rate):
                print(f"Skipping already evaluated config...")
                skip = True

        if not skip:
            print(f'{datetime.now()} [The only process]: Start training ({emb_dim}, {num_layers}, {n_heads}, {mlp_dims}, {dropout_rate})...', flush=True)

            acc, loss = train_model(model_name, dataset_source, cv_sets, {
                'num_layers': num_layers,
                'embedding_size': emb_dim,
                'num_heads': n_heads,
                'dropout_rate': dropout_rate,
                'num_channels': 1,
                'mlp_dim': mlp_dims,
                'patch_size': X[0].shape[1],
                'num_patches': X[0].shape[0],
            }, results_path)

            print(f'{datetime.now()} [The only process]: Done ({acc}, {loss})...', flush=True)

def tune_transformer_step_size(step_sizes):
    results = {}

    for step_size in step_sizes:
        X, y = build_dataset(step_size)

        cv_sets = create_cross_validation_sets(X, y, chunks=5)
        (X_train, y_train, X_val, y_val) = cv_sets[0]

        transformer = build_transformer(X_train.shape[2])
        transformer.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

        val_loss, val_accuracy = transformer.evaluate(X_val, y_val)

        results[step_size] = (val_loss, val_accuracy)

    losses = [results[s][0] for s in step_sizes]
    accuracies = [results[s][1] for s in step_sizes]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Step Size')
    ax1.set_ylabel('Validation Loss', color='tab:red')
    ax1.plot(step_sizes, losses, 'o-', color='tab:red', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='tab:blue')
    ax2.plot(step_sizes, accuracies, 's-', color='tab:blue', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Validation Loss and Accuracy by Step Size')
    fig.tight_layout()
    plt.show()

    return results

def print_results_table(path):
    if not path.endswith('.json'):
        path = f'models/{path}/results.json'

    if not os.path.exists(path):
        print(f"No results file found at {path}")
        return

    with open(path, "r") as f:
        results = json.load(f)

    if not results:
        print("No results to display.")
        return

    sorted_results = sorted(results, key=lambda x: x["val_loss"])[:25]

    headers = ["Embed", "Layers", "Heads", "MLP Dim", "Dropout rate", "Loss", "Acc", "Overfit Factor"]
    table = [
        [
            r["embedding_dim"],
            r["n_layers"],
            r["n_attn_heads"],
            r["mlp_dims"],
            r["dropout_rate"],
            round(r["val_loss"], 4),
            round(r["val_accuracy"], 4),
            round(r["overfit_factor"], 1)
        ]
        for r in sorted_results
    ]

    print(tabulate(table, headers=headers, tablefmt="grid"))


# print_results_table()
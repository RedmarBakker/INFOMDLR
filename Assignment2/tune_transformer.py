from textwrap import indent

import keras

import platform

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from keras.src.saving import load_model
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.python.distribute.multi_process_lib import multiprocessing

from models.transformer import ClassToken
from data import build_dataset
from data import create_cross_validation_sets
from models.transformer import build_transformer
import matplotlib.pyplot as plt
from itertools import product
import json
import os
from tabulate import tabulate
import numpy as np
from multiprocessing import Pool, Manager
from functools import partial

# for i, (emb_dim, num_layers, n_heads, mlp_dims, dropout_rate) in enumerate(combinations):
def train_model(dataset_source, cv_sets, config, results_path):
    # Build the Transformer model

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for i, (X_train, y_train, X_val, y_val) in enumerate(cv_sets):

        # Now use these values to build your model
        transformer = build_transformer(config)

        transformer.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callback filepath
        filepath = f"./models/transformer/{dataset_source}/emb{config['embedding_size']}_layers{config['num_layers']}_heads{config['num_heads']}_mlp{config['mlp_dim']}_dropout{config['dropout_rate']}_cv{i}.keras"

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=100,
            restore_best_weights=True
        )
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            #verbose=1
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.95,
            patience=1,
            min_lr=1e-6,
            #verbose=1
        )

        transformer.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val), batch_size=16, callbacks=[model_checkpoint, reduce_lr, early_stopping], verbose=0)

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

def process_worker(process_id, combination_stack, cv_sets, patch_size, num_patches, dataset_source, results_path):
    while combination_stack:
        try:
            (emb_dim, num_layers, n_heads, mlp_dims, dropout_rate) = combination_stack.pop()

            print(f'{datetime.now()} [Process {process_id+1}]: Start training ({emb_dim}, {num_layers}, {n_heads}, {mlp_dims}, {dropout_rate})...', flush=True)

            acc, loss = train_model(dataset_source, cv_sets, {
                'num_layers': num_layers,
                'embedding_size': emb_dim,
                'num_heads': n_heads,
                'dropout_rate': dropout_rate,
                'num_channels': 1,
                'mlp_dim': mlp_dims,
                'patch_size': patch_size,
                'num_patches': num_patches,
            }, results_path)

            print(f'{datetime.now()} [Process {process_id+1}]: Done ({acc}, {loss})...', flush=True)
        except IndexError:
            break

def tune_transformer_parameters(embedding_dims, n_layers, n_attn_heads, mlp_dims, dropout_rate, reversed_execution=False):
    with_multiprocessing = False

    # Process data
    dataset_source = 'intra'  # intra or 'cross'

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
    X, y = build_dataset(all_files, 1022)
    cv_sets = create_cross_validation_sets(X, y, chunks=n_folds)

    results_path = f"models/transformer/{dataset_source}/results.json"

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Ensure the results file exists
    if not os.path.exists(results_path):
        with open(results_path, 'w') as f:
            json.dump([], f)

    print(f'Storing results in: {results_path}')

    manager = Manager()

    configurations = list(product(embedding_dims, n_layers, n_attn_heads, mlp_dims, dropout_rate))

    if not reversed_execution:
        configurations = reversed(configurations)

    combination_stack = manager.list(configurations)

    if with_multiprocessing:
        n_processes = 2
        print(f'Starting {n_processes} processes...')

        with Pool(processes=os.cpu_count()) as pool:
            pool.starmap(
                process_worker,
                [(process_id, combination_stack, cv_sets, X[0].shape[1], X[0].shape[0], dataset_source, results_path)
                 for process_id in range(n_processes)]
            )
    else:
        process_worker(0, combination_stack, cv_sets, X[0].shape[1], X[0].shape[0], dataset_source, results_path)

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
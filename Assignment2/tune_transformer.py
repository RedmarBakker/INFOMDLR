import keras
from keras.src.saving import load_model
from sklearn.model_selection import train_test_split

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

def tune_transformer_parameters(embedding_dims, n_layers, n_attn_heads, mlp_dims, dropout_rate):
    # Process data
    dataset_source = 'cross'  # or 'cross'
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

    # Train test split
    # train_filepaths, val_filepaths = train_test_split(
    #     all_files,
    #     test_size=0.18,
    #     random_state=42
    # )

    # print('Num train files: ', len(train_filepaths))

    X, y = build_dataset(all_files, 1022)
    cv_sets = create_cross_validation_sets(X, y, chunks=5)

    results_path = f"models/transformer/{dataset_source}/results.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    combinations = list(product(embedding_dims, n_layers, n_attn_heads, mlp_dims, dropout_rate))

    for i, (emb_dim, num_layers, n_heads, mlp_dims, dropout_rate) in enumerate(combinations):
        # Build the Transformer model

        config = {}
        config['num_layers'] = num_layers
        config['embedding_size'] = emb_dim
        config['num_heads'] = n_heads
        config['dropout_rate'] = dropout_rate
        config['num_channels'] = 1
        config['mlp_dim'] = mlp_dims
        config['patch_size'] = X[0].shape[1]
        config['num_patches'] = X[0].shape[0]

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
            filepath = f"./models/transformer/{dataset_source}/emb{emb_dim}_layers{num_layers}_heads{n_heads}_mlp{mlp_dims}_dropout{dropout_rate}_cv{i}.keras"

            # Callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=80,
                restore_best_weights=True
            )
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.95,
                patience=1,
                min_lr=1e-6,
                verbose=1
            )

            transformer.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val), batch_size=16, callbacks=[model_checkpoint, reduce_lr, early_stopping])

            best_model = load_model(filepath, {'ClassToken': ClassToken})
            train_loss, train_accuracy = best_model.evaluate(X_train, y_train)
            val_loss, val_accuracy = best_model.evaluate(X_val, y_val)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_accuracy)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_accuracy)

        config_result = {
            "embedding_dim": emb_dim,
            "n_layers": num_layers,
            "n_attn_heads": n_heads,
            "mlp_dims": mlp_dims,
            "dropout_rate": dropout_rate,

            # "val_loss": val_loss,
            # "val_accuracy": val_accuracy,
            # "overfit_factor": val_loss / train_loss,
            "val_loss": np.mean(val_loss_list),
            "val_accuracy": np.mean(val_acc_list),
            "overfit_factor": np.mean(val_loss_list) / np.mean(train_loss_list),
        }

        all_results.append(config_result)

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

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

def print_results_table(path="tuning_results/transformer_parameters.json"):
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


print_results_table()
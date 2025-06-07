from keras.src.callbacks import EarlyStopping

from data import build_dataset, create_cross_validation_sets
from models.transformer import build_transformer
import matplotlib.pyplot as plt
from itertools import product
import json
import os
from tabulate import tabulate
import numpy as np

def tune_transformer_parameters(step_sizes, embedding_dims, n_layers, n_attn_heads, attn_dropouts, ffn_dropouts, final_units, final_dropouts):
    results_path = "tuning_results/transformer_parameters.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    combinations = list(product(step_sizes, embedding_dims, n_layers, n_attn_heads, attn_dropouts, ffn_dropouts, final_units, final_dropouts))

    for i, (step_size, emb_dim, num_layers, n_heads, attn_do, ffn_do, fdu, fdo) in enumerate(combinations):
        X, y = build_dataset(step_size)

        cv_sets = create_cross_validation_sets(X, y, chunks=8)

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        for (X_train, y_train, X_val, y_val) in cv_sets:
            # Now use these values to build your model
            transformer = build_transformer(
                data_length=X.shape[2],
                embedding_dim=emb_dim,
                n_layers=num_layers,
                n_attn_heads=n_heads,
                attn_dropout=attn_do,
                ffn_dropout=ffn_do,
                final_dense_units=fdu,
                final_dropout=fdo
            )

            transformer.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[])

            train_loss, train_accuracy = transformer.evaluate(X_train, y_train)
            val_loss, val_accuracy = transformer.evaluate(X_val, y_val)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_accuracy)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_accuracy)

        config_result = {
            "step_size": step_size,
            "embedding_dim": emb_dim,
            "n_layers": num_layers,
            "n_attn_heads": n_heads,
            "attn_dropout": attn_do,
            "ffn_dropout": ffn_do,
            "final_dense_units": fdu,
            "final_dropout": fdo,
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

    headers = ["Step", "Embed", "Layers", "Heads", "Attn DO", "FFN DO", "Dense", "Final DO", "Loss", "Acc", "Overfit Factor"]
    table = [
        [
            r["step_size"],
            r["embedding_dim"],
            r["n_layers"],
            r["n_attn_heads"],
            r["attn_dropout"],
            r["ffn_dropout"],
            r["final_dense_units"],
            r["final_dropout"],
            round(r["val_loss"], 4),
            round(r["val_accuracy"], 4),
            round(r["overfit_factor"], 1)
        ]
        for r in sorted_results
    ]

    print(tabulate(table, headers=headers, tablefmt="grid"))


print_results_table()
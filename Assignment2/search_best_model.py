import os
import tensorflow as tf
import keras
import numpy as np
from models.transformer import ClassToken
from sklearn.model_selection import train_test_split
from data import build_dataset

model_folder = './models/transformer/cross/best_models/'

filepath = './Intra/train/'

all_files = []
for root, _, files in os.walk(filepath):
    for file in files:
        if file.endswith('.h5'):
            all_files.append(os.path.join(root, file))

all_files.sort()
print(f"Found {len(all_files)} files in {filepath}")

X, y = build_dataset(all_files)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


print(f"Created train and validation sets with {len(X_train)} and {len(X_val)} samples respectively.")

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
            loss, accuracy = model.evaluate(X_val, y_val, verbose=1)

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





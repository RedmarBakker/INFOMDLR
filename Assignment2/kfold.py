import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from data import build_dataset
from visualization import plot_dataset_as_meg, plot_dataset_as_lines
from models.autoencoder import build_autoencoder, chunk_data, glue_chunks
from models.transformer import build_transformer
from models.spat_temp import build_spat_temp_model, spat_lstm

n_splits = 4 # Given 32 samples, 4 splits means 24 train, 8 val
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_accuracies = []
X, y = build_dataset()


for fold_idx, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f"\n--- FOLD {fold_idx+1}/{n_splits} ---")
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        model_fold = build_spat_temp_model()


        print(f"Training on {len(X_train_fold)} samples, validating on {len(X_val_fold)} samples.")
        history = model_fold.fit(
            X_train_fold, y_train_fold,
            epochs=20, # Might need more or less, use EarlyStopping
            batch_size=4, # Small batch size
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)],
            verbose=0 # Set to 1 to see epoch progress for the first fold, then 0
        )

        val_loss, val_acc = model_fold.evaluate(X_val_fold, y_val_fold, verbose=0)
        print(f"Fold {fold_idx+1} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        fold_accuracies.append(val_acc)
        if fold_idx == 0: # See detailed logs for first fold then suppress for brevity
             keras.utils.plot_model(model_fold, show_shapes=True, to_file=f'spatial_lstm_model_fold{fold_idx+1}.png')


print("\n--- Cross-Validation Results ---")
print(f"Mean Validation Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
# (a) Choose a suitable deep learning model for the involved classification tasks. Justify your choice.
# (b) Compare the accuracy of the 2 types of classification, i.e. intra-subject and cross subject data using your model. Explain your results.
# (c) Explain the choices of hyper-parameters of your model architecture and analyze their influence on the results (for both 2 types of classification). How they are selected?
# (d) If there is a significant difference in training and testing accuracies, what could be a possible reason? What are the alternative models or approaches you would select? Select one and implement to further improve your results. Justify your choice.
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib
# Use the TkAgg backend for matplotlib to avoid the need for a display
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


from tuning import tune_transformer_parameters
from data import build_dataset, create_cross_validation_sets, z_norm, load, rest_set_names, motor_set_names, math_set_names, memory_set_names
from visualization import plot_dataset_as_meg, plot_dataset_as_lines
from models.autoencoder import build_autoencoder, chunk_data, glue_chunks, chunk_data_for_conv2d
from models.transformer import build_transformer


# plot_dataset_as_lines(z_norm(load(rest_set_names[0])))
# plot_dataset_as_meg(load(rest_set_names[0]))
# plot_dataset_as_meg(load(motor_set_names[0]))
# plot_dataset_as_meg(load(math_set_names[0]))
# plot_dataset_as_meg(load(memory_set_names[0]))


# X, y = build_dataset()
# cv_sets = create_cross_validation_sets(X, y, chunks=5)
# (X_train, y_train, X_val, y_val) = cv_sets[0]
#
# # (32, 248, 35624)
# print(X.shape)
#
# # sample length = 35624
# transformer = build_transformer(X.shape[2])

step_sizes = [525, 550, 575, 625, 650, 675]


# { 100: (4.42, 0.33), 500: (5.775, 0.5), 1000: (5.08, 0), 2000: (8.17, 0)}
# {200: (6.33782958984375, 0.0), 300: (6.816913604736328, 0.0), 400: (4.2272467613220215, 0.1666666716337204), 600: (3.165964126586914, 0.5), 700: (4.3197174072265625, 0.1666666716337204), 800: (4.336094856262207, 0.0), 900: (4.497375011444092, 0.1666666716337204)}
# {525: (1.5875426530838013, 0.6666666865348816), 550: (5.109252452850342, 0.0), 575: (8.496978759765625, 0.0), 625: (4.025796413421631, 0.3333333432674408), 650: (6.950931072235107, 0.3333333432674408), 675: (3.791290283203125, 0.0)}

tune_transformer_parameters(
    [25],
    [16, 24],
    [2],
    [2,4,5],
    [0.3],
    [0.3],
    [32],
    [0.3],
)

# all_val_accuracies = []
#
# for fold_id, (X_train, y_train, X_val, y_val) in enumerate(cv_sets):
#     print(f"Fold {fold_id + 1}")
#
#     # Rebuild a fresh transformer for each fold (to avoid weight reuse)
#     model = build_transformer(X.shape[2]) # input_dim = time length per sample
#
#     # Fit on training data
#     model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
#
#     # Optional: evaluate and collect validation accuracy
#     val_loss, val_accuracy = model.evaluate(X_val, y_val)
#     all_val_accuracies.append(val_accuracy)
#
# print("Validation accuracies across folds:", all_val_accuracies)
# print("Mean validation accuracy:", np.mean(all_val_accuracies))
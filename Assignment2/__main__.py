# (a) Choose a suitable deep learning model for the involved classification tasks. Justify your choice.
# (b) Compare the accuracy of the 2 types of classification, i.e. intra-subject and cross subject data using your model. Explain your results.
# (c) Explain the choices of hyper-parameters of your model architecture and analyze their influence on the results (for both 2 types of classification). How they are selected?
# (d) If there is a significant difference in training and testing accuracies, what could be a possible reason? What are the alternative models or approaches you would select? Select one and implement to further improve your results. Justify your choice.
import numpy as np
from sklearn.model_selection import train_test_split
import os
import keras
import math
import json
import matplotlib
from itertools import chain

from tune_transformer import print_results_table
from tune_transformer import tune_transformer_parameters

# Use the TkAgg backend for matplotlib to avoid the need for a display
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from data import z_norm, load, load_files, extract_patches, TransformerDataGenerator, glue_chunks, CustomAccuracyLossCheckpoint
from visualization import plot_dataset_as_meg, plot_dataset_as_lines
from models.autoencoder import build_autoencoder, chunk_data, chunk_data_for_conv2d
from models.transformer import build_transformer
from models.spat_temp import build_spat_temp_model


# plot_dataset_as_lines(z_norm(load(rest_set_names[0])))
# plot_dataset_as_meg(load(rest_set_names[0]))
# plot_dataset_as_meg(load(motor_set_names[0]))
# plot_dataset_as_meg(load(math_set_names[0]))
# plot_dataset_as_meg(load(memory_set_names[0]))

# tune_transformer_parameters(
#     [32, 64, 128],
#     [2, 4, 6],
#     [2, 4, 6],
#     [64, 128],
#     [0.3],
#     reversed_execution=True,
#     with_autoencoder=True
# )

print_results_table('transformer_autoencoder/cross')
# print_results_table('transformer/cross')

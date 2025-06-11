# (a) Choose a suitable deep learning model for the involved classification tasks. Justify your choice.
# (b) Compare the accuracy of the 2 types of classification, i.e. intra-subject and cross subject data using your model. Explain your results.
# (c) Explain the choices of hyper-parameters of your model architecture and analyze their influence on the results (for both 2 types of classification). How they are selected?
# (d) If there is a significant difference in training and testing accuracies, what could be a possible reason? What are the alternative models or approaches you would select? Select one and implement to further improve your results. Justify your choice.
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
# Use the TkAgg backend for matplotlib to avoid the need for a display
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from data import load_files, DataGenerator
from visualization import plot_dataset_as_meg, plot_dataset_as_lines
from models.autoencoder import build_autoencoder, chunk_data, glue_chunks, chunk_data_for_conv2d
from models.transformer import build_transformer
from models.spat_temp import build_spat_temp_model


# plot_dataset_as_lines(z_norm(load(rest_set_names[0])))
# plot_dataset_as_meg(load(rest_set_names[0]))
# plot_dataset_as_meg(load(motor_set_names[0]))
# plot_dataset_as_meg(load(math_set_names[0]))
# plot_dataset_as_meg(load(memory_set_names[0]))

model = 'cross' # or 'intra'
match model:
    case 'intra':
        filepath = './intra/train'
    case 'cross':
        filepath = './cross/train'
    case _:
        raise ValueError(f"Unknown training set: {model}")


# Train-test split
train_files, val_files = train_test_split(load_files(filepath),test_size=0.18,
                                          random_state=42)

# ---- TRAINING ----
# Steps (blijf een stap toepassen tot het model underfit,
# daarna naar de volgende stap:
# paper model --> overfits despite regularization (0.5 dropout, l2 X, TA: 1, VA: .25)
# Misschien moet je soms de dropout verlagen als je het model minder complex maakt.
# TODO decrease model complexity (in the following order):
# 1. decreasing temporal blocks (3->2->etc.)
# 2. reduce the number of filters (16->12->8-> etc.)
# 3. reduce the number of residual blocks (4->2->etc.)


frequencies = [500, 250]
for freq in frequencies:

    # hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 32
    DOWNSAMPLE_FREQ = freq

    # Data variables
    train_generator = DataGenerator(filepaths=train_files,
                                    batch_size=BATCH_SIZE,
                                    new_sfreq=DOWNSAMPLE_FREQ)
    val_generator = DataGenerator(filepaths=val_files,
                                    batch_size=BATCH_SIZE,
                                    new_sfreq=DOWNSAMPLE_FREQ)
    print(f"Split with downsample frequency: {DOWNSAMPLE_FREQ}Hz")

    # Build model
    input_shape = train_generator[0][0][0].shape
    cnn = build_spat_temp_model(train_generator[0][0][0].shape, embed_dim=16, dropout=0.5)
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(f"\nStarting training using generator. Batches per training epoch: {len(train_generator)}")
    print(f"Batches per validation epoch: {len(val_generator)}")
    print(f"Model expected input shape: {cnn.input_shape}")

    history = cnn.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        verbose=1,
    )

    print("\nClassifier training complete!")

    # --- Plot training history ---
    plt.plot(history.history['accuracy'], label='Training Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Classifier Training History with {freq}Hz (Cross-Subject)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()




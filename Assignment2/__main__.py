# (a) Choose a suitable deep learning model for the involved classification tasks. Justify your choice.
# (b) Compare the accuracy of the 2 types of classification, i.e. intra-subject and cross subject data using your model. Explain your results.
# (c) Explain the choices of hyper-parameters of your model architecture and analyze their influence on the results (for both 2 types of classification). How they are selected?
# (d) If there is a significant difference in training and testing accuracies, what could be a possible reason? What are the alternative models or approaches you would select? Select one and implement to further improve your results. Justify your choice.
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

# plot_dataset_as_lines(load(rest_set_names[0]))
# plot_dataset_as_meg(load(rest_set_names[0]))
# plot_dataset_as_meg(load(motor_set_names[0]))
# plot_dataset_as_meg(load(math_set_names[0]))
# plot_dataset_as_meg(load(memory_set_names[0]))

X, y = build_dataset()
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,       # Or your desired validation fraction
    stratify=y  # IMPORTANT: to maintain class proportions in train and val sets
)
# transformer = build_transformer(X.shape[2])

# transformer.fit(X, y)


model = build_spat_temp_model(X_train[0].shape, dropout=0.3, l2fac=0.001)
history = model.fit(X_train,y_train,batch_size=32,epochs=100,verbose=1, validation_data=(X_val,y_val))#,callbacks=[keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)])

model_filepath = 'spat_temp_model.keras' # Using the .keras extension

# Save the model
model.save(model_filepath)

print(history.history.keys()) # Should include 'accuracy' and 'loss', and 'val_accuracy', 'val_loss' if validation_data was used


# Plot training accuracy
plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# # Plot training loss
# plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
# plt.plot(history.history['loss'], label='Training Loss')
# if 'val_loss' in history.history:
#     plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

plt.tight_layout() # Adjust subplots to fit in figure area.
plt.show()

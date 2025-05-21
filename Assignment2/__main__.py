# (a) Choose a suitable deep learning model for the involved classification tasks. Justify your choice.
# (b) Compare the accuracy of the 2 types of classification, i.e. intra-subject and cross subject data using your model. Explain your results.
# (c) Explain the choices of hyper-parameters of your model architecture and analyze their influence on the results (for both 2 types of classification). How they are selected?
# (d) If there is a significant difference in training and testing accuracies, what could be a possible reason? What are the alternative models or approaches you would select? Select one and implement to further improve your results. Justify your choice.

from data import load, rest_set_names, motor_set_names, memory_set_names, math_set_names
from visualization import plot_dataset_as_meg, plot_dataset_as_lines
from models.autoencoder import build_autoencoder, chunk_data, glue_chunks
from models.transformer import build_transformer

# plot_dataset_as_lines(load(rest_set_names[0]))
# plot_dataset_as_meg(load(rest_set_names[0]))
# plot_dataset_as_meg(load(motor_set_names[0]))
# plot_dataset_as_meg(load(math_set_names[0]))
# plot_dataset_as_meg(load(memory_set_names[0]))

rest1 = load(rest_set_names[0])

build_transformer().summary()




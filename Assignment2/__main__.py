# (a) Choose a suitable deep learning model for the involved classification tasks. Justify your choice.
# (b) Compare the accuracy of the 2 types of classification, i.e. intra-subject and cross subject data using your model. Explain your results.
# (c) Explain the choices of hyper-parameters of your model architecture and analyze their influence on the results (for both 2 types of classification). How they are selected?
# (d) If there is a significant difference in training and testing accuracies, what could be a possible reason? What are the alternative models or approaches you would select? Select one and implement to further improve your results. Justify your choice.


import h5py

def get_dataset_name(filename_with_dir):
    filename_without_dir = filename_with_dir.split('/')[-1]
    temp = filename_without_dir.split('.')[:-1]
    dataset_name = ''.join(temp)

    return dataset_name

filename_path = "Intra/train/rest1059231.h5"

with h5py.File(filename_path, 'r') as f:
    dataset_name = get_dataset_name(filename_path)
    matrix = f.get(dataset_name)[()]

    print(type(matrix))
    print(matrix.shape)


import numpy as np
import pandas as pd

def load_data(filename):
    file_data = np.loadtxt(filename, delimiter=',')

    if file_data.dtype == np.dtype('O'):  # Check if data type is object (text data)
        data_temp = file_data[:, 1:]
        labels_temp = np.array(file_data[:, 0])
    else:
        data_temp = file_data[:, 1:]
        labels_temp = file_data[:, 0]

    data = data_temp
    labels = labels_temp

    return data, labels

def load_base(filename):
    file_data = pd.read_csv(filename, header=None)

    array_x = pd.DataFrame(file_data.iloc[:, 1:].values) # Todas as colunas, exceto a primeira
    array_y = pd.DataFrame(file_data.iloc[:, 0].values) # A primeira coluna

    return array_x, array_y
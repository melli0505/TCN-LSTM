import csv
import os
import numpy as np
import scipy

def get_data(dir:str, training:bool=False):
    """
    Get current data to make prediction.

    Args:
        dir (str): Data source directory
    
    Returns:
        data (np.array): Numpy-array data, shape=(None, 3200)
    """
    file_list = os.listdir(dir)
    if not training:
        file_list = sorted(file_list, reverse=True)
        i = -1
        for filename in file_list:
            if os.path.isdir(dir + filename): continue
        data = []
        file = open(dir + filename, "r")
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            data.extend(row)
        data = np.array(data, dtype=np.float32).reshape(len(data) // 800, 800)
        return data
    else:
        classes = os.listdir(dir)
        training_data = []
        training_label = []
        for c in range(len(classes)):
            file_list = os.listdir(dir + "/" + classes[c])
            i = -1
            data = []
            for filename in file_list:
                if os.path.isdir(dir + "/" + classes[c] + "/" + filename): continue
                file = open(dir + "/" + classes[c] + "/" + filename, "r")
                reader = csv.reader(file, delimiter=',')
                for row in reader:
                    data.extend(row)
            training_data.append(np.array(data, dtype=np.float32).reshape(len(data) // 800, 800))
            training_label.append(np.array([[0 if i != c else 1 for i in range(len(classes))] for _ in range(len(data) // 800)]))

        training_data = np.vstack(training_data)
        training_label = np.vstack(training_label)
        return training_data, training_label
        
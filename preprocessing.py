import csv
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def get_data(dir: str, scaler_data_path: str, training: bool = False):
    """
    Get current data to make prediction.

    Args:
        dir (str): Data source directory

    Returns:
        data (np.array): Numpy-array data, shape=(None, 3200)
    """
    nor = []
    normal1 = csv.reader(open(scaler_data_path, "r"), delimiter=",")
    for row in normal1:
        nor.extend(row)
    nor = np.array(nor, dtype=np.float32)

    scaler = MinMaxScaler()
    scaler.fit(nor.reshape(-1, 1))

    file_list = os.listdir(dir)
    if not training:
        file_list = sorted(file_list, reverse=True)
        i = -1
        for filename in file_list:
            if os.path.isdir(dir + filename):
                continue
        data = []
        file = open(dir + filename, "r")
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            data.extend(row)
        data = scaler.transform(np.array(data).reshape(102400, 1))
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
                if os.path.isdir(dir + "/" + classes[c] + "/" + filename):
                    continue
                file = open(dir + "/" + classes[c] + "/" + filename, "r")
                reader = csv.reader(file, delimiter=",")
                for row in reader:
                    data.extend(row)

            data = np.array(data, dtype=np.float32).reshape(len(data), 1)
            data = scaler.transform(data)
            data = data.reshape(len(data) // 800, 800)
            training_data.append(data)
            training_label.append([c for _ in range(len(data))])

        training_data = np.vstack(training_data)
        training_label = np.hstack(training_label)

        print(training_data.shape, training_label.shape)
        return training_data, training_label

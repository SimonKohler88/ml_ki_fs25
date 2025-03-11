#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import os

_path = os.path.dirname(os.path.abspath(__file__))
_archive = os.path.join(_path, 'archive')

FishName_to_ClassNumber = {
    'barrel_jellyfish': 0,
    'blue_jellyfish': 1,
    'compass_jellyfish': 2,
    'lions_mane_jellyfish': 3,
    'mauve_stinger_jellyfish': 4,
    'Moon_jellyfish': 5
}

ClassNumber_to_FishName = {
    0: 'barrel_jellyfish',
    1: 'blue_jellyfish',
    2: 'compass_jellyfish',
    3: 'lions_mane_jellyfish',
    4: 'mauve_stinger_jellyfish',
    5: 'Moon_jellyfish'
}

def _load_walking(path):
    # walks recursively trough given directory
    # returns a list [[data1, data2, ...],[foldername_data1, foldername_data2, ...]]
    # whereas data1, data2, ... are the loaded .npy files (numpy arrays)
    # must be previously be converted.
    x_data = []
    y_data = []
    for directory, folder, files in os.walk(path):
        if len(folder) > 0:
            continue
        # check which jellyfish
        fish = os.path.split(directory)[-1]

        # append to array [[array1, array2, ...], [fish1, fish2, ...]]
        for f in files:
            file, ext = os.path.splitext(f)
            if ext == '.npy':  # only use numpy representations
                file_path = os.path.join(directory, f)
                raw_data = np.load(file_path)
                # print(raw_data.shape)
                x_data.append(raw_data)
                y_data.append(FishName_to_ClassNumber[fish])
    y_data = np.array(y_data)
    return [np.asarray(x_data), y_data]


def load_train_test_from_np():
    """ Loads Jellyfish Data from Train_Test_Valid folder

    :return:
    """
    path_test = os.path.join(_archive, 'Train_Test_Valid', 'test')
    path_train = os.path.join(_archive, 'Train_Test_Valid', 'Train')

    train_data = _load_walking(path_train)
    test_data = _load_walking(path_test)
    return train_data, test_data


if __name__ == '__main__':
    train, test = load_train_test_from_np()

    print(train[0].shape)
    print(test[0].shape)

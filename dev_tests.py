#!/usr/bin/python3
# -*- coding: utf-8 -*-


from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file = 'archive/barrel_jellyfish/01.npy'
    raw_data = np.load(file)

    model = keras.Sequential(
        [
            keras.Input(shape=(179, 179, 3)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")

        ]
    )

    # y = keras.layers.Conv2D(32, 3, activation='relu')(raw_data)
    x = keras.Input(shape=(179, 179, 3))
    y = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(),

    plt.figure()
    plt.imshow(y)
    plt.show()
    print(y.shape)

#!/usr/bin/python3
# -*- coding: utf-8 -*-


from tensorflow import keras

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print(x_test[0], y_test)

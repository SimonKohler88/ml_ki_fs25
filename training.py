# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)

    # Daten Aufbereitung
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Model definition
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    # https://keras.io/api/layers/convolution_layers/convolution2d/
    # https://keras.io/api/layers/pooling_layers/max_pooling2d/
    # https://keras.io/api/layers/reshaping_layers/flatten/
    # https://keras.io/api/layers/regularization_layers/dropout/
    # https://keras.io/api/layers/core_layers/dense/

    model.summary()

    batch_size = 128
    epochs = 15
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    laden = False  # True: vorher erstelltes Modell laden; False: neues Modell fitten
    if laden:
        model = keras.models.load_model('mymodel')
    else:
        # https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.grid()
        plt.show()
        # summarize history for loss
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.grid()
        plt.show()

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    y_pred = model.predict(x_test)
    y_pred_idx = np.argmax(y_pred, axis=1)
    y_test_idx = np.argmax(y_test, axis=1)
    sum(y_pred_idx - y_test_idx != 0)
    N = 1000  # x_test.shape[0]
    for k in range(N):
        if (y_test_idx[k] != y_pred_idx[k]):
            print("Fehler - korrekt: ", y_test_idx[k], ", vorhergesagt: ", y_pred_idx[k])
            plt.figure()
            plt.imshow(x_test[k], cmap=plt.get_cmap('gray'))
            plt.title("Fehler - korrekt: " + str(y_test_idx[k]) + ", vorhergesagt: " + str(y_pred_idx[k]))
            plt.show()
            # input("Bitte Enter drücken...")
            # plt.close()

    model.save('./mymodel/mymodel.keras')
    # https://www.tensorflow.org/guide/keras/save_and_serialize

    for layer in model.layers:
        weights = layer.get_weights()
        print("Layer ", layer)
        print(weights)

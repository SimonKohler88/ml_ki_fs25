 #-*- coding: utf-8 -*-

import tensorflow as tf
import platform

from keras.src.layers import Dropout
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
import numpy as np
from load_jellyfish_data import load_train_test_from_np, ClassNumber_to_FishName
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Model / data parameters
model_save_path = './mymodel/mymodel.keras'
num_classes = int(len(ClassNumber_to_FishName))
input_shape = (179, 179, 3)
batch_size = 20
epochs = 60

if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)

    # Daten //validation hinzugefügt
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = load_train_test_from_np()
    # (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_val = x_val.astype("float32") / 255

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    print(x_val.shape[0], "validation samples")

    # convert class vectors to binary class matrices, one hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    # Data Augmentation

    datagen = ImageDataGenerator(
        rotation_range=30,  # Rotate images up to 15° (instead of 40°)
        width_shift_range=0.1,  # Shift images slightly (instead of 30%)
        height_shift_range=0.1,
        horizontal_flip=True,  # Flip images horizontally
        vertical_flip=True,  # Flip images vertically
        zoom_range=0.3,  # Small zooming (instead of 30%)
        shear_range=0.1,  # Small shear distortion (instead of 30%)
       # channel_shift_range=0.1,
        fill_mode="nearest"
    )

    # Apply augmentation only to training data
    datagen.fit(x_train)

    # Model definition
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),

            # First Conv Block many filters, large kernel for shape and global contrast
            layers.Conv2D(32, (3, 3), 1, padding="same", activation="relu", input_shape=input_shape),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Second Conv Block 3x3 kernel for finer structures such as tentacles
            layers.Conv2D(64, (3, 3), 1, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            #layers.Dropout(0.3),

            # Dense Layers
            layers.Dense(128, activation="relu"),
            layers.Dense(6, activation="softmax")
        ]
    )

    # https://keras.io/api/layers/convolution_layers/convolution2d/
    # https://keras.io/api/layers/pooling_layers/max_pooling2d/
    # https://keras.io/api/layers/reshaping_layers/flatten/
    # https://keras.io/api/layers/regularization_layers/dropout/
    # https://keras.io/api/layers/core_layers/dense/

    model.summary()

    # Learning rate scheduling
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.0001,  # Start with 0.0001
        decay_steps=100,  # Decay every 100 steps
        decay_rate=0.98,  # Reduce by 4% each step
        staircase=True  # Reduce in steps, not continuously
    )

    if platform.system() == "Darwin" and platform.processor() == "arm":
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Apply learning rate schedule to Adam optimizer
    model.compile(opt, loss="categorical_crossentropy", metrics=["accuracy"])
    # model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    laden = False  # True: vorher erstelltes Modell laden; False: neues Modell fitten
    if laden:
        model = keras.models.load_model(model_save_path)

    else:
        # https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class
        # history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        # Early Stopping hinzufügen
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True, verbose=1)

        # Modelltraining mit Early Stopping
        train_generator = datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
        validation_generator = datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

        #history1 = model.fit(train_generator, epochs=epochs,
                           # validation_data=validation_generator,
                            #callbacks=[early_stopping])

        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=(x_val, y_val),  # Correct validation set, no augmentation
            callbacks=[early_stopping]
        )

        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
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
    N = x_test.shape[0]
    for k in range(N):
        if (y_test_idx[k] != y_pred_idx[k]):
            print("Fehler - korrekt: ", y_test_idx[k], ", vorhergesagt: ", y_pred_idx[k])
            # plt.figure()
            # plt.imshow(x_test[k], cmap=plt.get_cmap('gray'))
            # plt.title("Fehler - korrekt: " + str(y_test_idx[k]) + ", vorhergesagt: " + str(y_pred_idx[k]))
            # plt.show()

            # input("Bitte Enter drücken...")
            # plt.close()

    # Compute confusion matrix
    cm = confusion_matrix(y_test_idx, y_pred_idx)

    # Optional: define class names
    class_names = [ClassNumber_to_FishName[i] for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

    # Optional: make sure everything aligns well
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title("Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    model.save(model_save_path)

    # too much data
    # https://www.tensorflow.org/guide/keras/save_and_serialize
    #
    # for layer in model.layers:
    #     weights = layer.get_weights()
    #     print("Layer ", layer)
    #     print(weights
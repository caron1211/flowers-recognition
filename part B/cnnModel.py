from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np


# build model
def buildModel(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))  # 5 for every type

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.summary()

    # Callback
    earlystop = EarlyStopping(patience=10)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    callbacks = [earlystop, learning_rate_reduction]

    return model, callbacks


def load_trained_model(weights_path, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS):
    load_model, _ = buildModel(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    load_model.load_weights(weights_path)
    return load_model

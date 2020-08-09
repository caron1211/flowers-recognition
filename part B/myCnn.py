from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import cnnModel
import prepareForCnn

FAST_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

path_train = "input/train"
path_test = "input/test"
batch_size = 15


def trainModel():
    train_df, validate_df = prepareForCnn.prepareTrain(path_train)
    prepareForCnn.showDivision(train_df)

    # Build model
    model, callbacks = cnnModel.buildModel(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]

    train_generator = getTrainGenerator(train_df)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df,
        path_train,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    #  fit model
    epochs = 3 if FAST_RUN else 50
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate // batch_size,
        steps_per_epoch=total_train // batch_size,
        callbacks=callbacks
    )

    # save model
    model.save_weights("model.h5")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1, epochs, 1))
    ax1.set_yticks(np.arange(0, 1, 0.1))

    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    ax2.set_xticks(np.arange(1, epochs, 1))

    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()


def loadModel():
    path_load = "model.h5"
    model = cnnModel.load_trained_model(path_load, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    return model


def testModel(model):
    print("test:")
    #  Prepare Testing Data
    test_df = prepareForCnn.prepareTest(path_test)
    nb_samples = test_df.shape[0]

    train_df, _ = prepareForCnn.prepareTrain(path_train)

    # Create Testing Generator
    test_gen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        path_test,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )
    y_true = prepareForCnn.convertToBinary(test_df['category'])

    predict = model.predict(test_generator, steps=np.ceil(nb_samples / batch_size))
    loss, acc = model.evaluate(test_generator, steps=np.ceil(nb_samples / batch_size))
    print("accuracy", acc)
    print("loss", loss)

    test_df['category'] = np.argmax(predict, axis=-1)

    train_generator = getTrainGenerator(train_df)

    label_map = dict((v, k) for k, v in train_generator.class_indices.items())
    test_df['category'] = test_df['category'].replace(label_map)

    # replace to numbers
    test_df = prepareForCnn.convertToBinary(test_df)

    prepareForCnn.showDivision(test_df)

    print(classification_report(y_true, test_df['category']))
    matrix = confusion_matrix(y_true, test_df['category'])
    print(matrix)


def getTrainGenerator(train_df):

    #  Traning Generator
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        path_train,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    return train_generator


if __name__ == "__main__":
    # trainModel()
    load_model = loadModel()

    # for layer in load_model.layers: print(layer.get_config())
    testModel(load_model)

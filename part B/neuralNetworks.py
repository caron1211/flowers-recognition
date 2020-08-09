from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras

import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical

import PrepareData

path_to_data_train = 'output/data.h5'
path_to_labels_train = 'output/labels.h5'
path_to__data_test = 'output/data_test.h5'
path_to_labels_test = 'output/labels_test.h5'

global_features, global_labels = PrepareData.import_feature_matrix(path_to_data_train, path_to_labels_train)
(train_features, valid_features, train_labels, valid_labels) = train_test_split(global_features,
                                                                                global_labels,
                                                                                test_size=0.25,shuffle=True)

test_features, test_labels = PrepareData.import_feature_matrix(path_to__data_test,path_to_labels_test)
test_features, test_labels = shuffle(test_features, test_labels)

model = keras.Sequential([
    keras.layers.Dense(100, input_dim=532, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

es_callback = keras.callbacks.EarlyStopping(patience=10)


model.fit(train_features, to_categorical(train_labels), validation_data=(valid_features, to_categorical(valid_labels)),
          epochs=200, callbacks=[es_callback])

test_loss, test_acc = model.evaluate(test_features, to_categorical(test_labels))
print('\nTest accuracy:', test_acc)
prediction = model.predict(test_features)
prediction = np.argmax(prediction, axis=-1)

print(classification_report(test_labels, prediction))
matrix = confusion_matrix(test_labels, prediction)
print(matrix)
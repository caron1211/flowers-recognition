import os

import cv2
import h5py
import mahotas as mahotas
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle

path_to_data_train = 'output/data.h5'
path_to_labels_train = 'output/labels.h5'
path_to__data_test = 'output/data_test.h5'
path_to_labels_test = 'output/labels_test.h5'

SIZE = (500, 500)
bins = 8


def get_image(row_id, root):
    """
    Converts an image number into the file path where the image is located,
    opens the image, resize and returns the image as a numpy array.
    """
    filename = row_id
    file_path = os.path.join(root, filename)
    img = cv2.imread(file_path)
    img = cv2.resize(img, SIZE)
    return img


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


def create_feature_matrix(path):
    # loop over the training data sub-
    filenames = os.listdir(path)
    global_features = []
    labels = []
    for filename in filenames:
        category = filename.split('_')[0]
        img = get_image(filename, path)
        fv_hu_moments = fd_hu_moments(img)
        fv_haralick = fd_haralick(img)
        fv_histogram = fd_histogram(img)
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        labels.append(category)
        global_features.append(global_feature)

    return global_features, labels


def save_feature_matrix(global_features, labels, path_to_save_data, path_to_save_labels):
    # encode the target labels
    targetNames = np.unique(labels)
    le = LabelEncoder()
    target = le.fit_transform(labels)

    # scale features in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)

    # save the feature vector using HDF5
    h5f_data = h5py.File(path_to_save_data, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

    h5f_label = h5py.File(path_to_save_labels, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()


def import_feature_matrix(path_to_save_data, path_to_save_labels):
    h5f_data = h5py.File(path_to_save_data, 'r')
    h5f_label = h5py.File(path_to_save_labels, 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()
    return global_features, global_labels


def createData():
    path_train = 'input/train'
    path_test = 'input/test'
    train_features, train_labels = create_feature_matrix(path_train)
    save_feature_matrix(train_features, train_labels, path_to_data_train, path_to_labels_train)
    print(len(train_features))
    print(len(train_features[0]))

    test_features, test_labels = create_feature_matrix(path_test)
    save_feature_matrix(test_features, test_labels, path_to__data_test, path_to_labels_test)


def getData():
    path_to_data_train = 'output/data.h5'
    path_to_labels_train = 'output/labels.h5'
    path_to__data_test = 'output/data_test.h5'
    path_to_labels_test = 'output/labels_test.h5'

    train_features, train_labels = import_feature_matrix(path_to_data_train, path_to_labels_train)
    train_features, train_labels = shuffle(train_features, train_labels)

    test_features, test_labels = import_feature_matrix(path_to__data_test, path_to_labels_test)
    test_features, test_labels = shuffle(test_features, test_labels)

    return train_features, train_labels, test_features, test_labels

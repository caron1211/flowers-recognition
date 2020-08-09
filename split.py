import shutil
import os
import numpy as np


def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)


def changeFileNames(path_to_data, name):

    for count, filename in enumerate(os.listdir(path_to_data)):
        dst = name + "_" + str(count) + ".jpg"
        src = path_to_data + filename
        dst = path_to_data + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)


def splitDirs(path_to_data, path_to_test_data, train_ratio):
    # get dirs
    _, dirs, _ = next(os.walk(path_to_data))
    # calculates how many train data per class
    data_counter_per_class = np.zeros((len(dirs)))
    for i in range(len(dirs)):
        path = os.path.join(path_to_data, dirs[i])
        files = get_files_from_folder(path)
        data_counter_per_class[i] = len(files)
        test_counter = np.round(data_counter_per_class * (1 - train_ratio))
    # transfers files
    for i in range(len(dirs)):
        path_to_original = os.path.join(path_to_data, dirs[i])
        path_to_save = os.path.join(path_to_test_data, dirs[i])
        # creates dir
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        files = get_files_from_folder(path_to_original)
        # moves data
        for j in range(int(test_counter[i])):
            dst = os.path.join(path_to_save, files[j])
            src = os.path.join(path_to_original, files[j])
            shutil.move(src, dst)


if __name__ == "__main__":
    path = "C:/Users/caron/Desktop/flower recognetion/kaggle flowers-recognition/kaggle flowers-recognition2/flowers/"
    # changeFileNames(path, "daisy")
    splitDirs(path, "input/new", 0.8)
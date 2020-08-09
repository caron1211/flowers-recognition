import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


def getInput(path):
    filenames = os.listdir(path)
    categories = []
    for filename in filenames:
        category = filename.split('_')[0]
        if category == 'flower':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    df["category"] = convertToString(df["category"])


    return df


def prepareTrain(path):

    df = getInput(path)

    # print(df.head())
    # sample = random.choice(filenames)
    # sample = random.choice(filenames)
    # image = load_img(path+sample)
    # plt.imshow(image)
    # plt.show()

    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    return train_df, validate_df


def showDivision(df):
    df['category'].value_counts().plot.bar()
    plt.show()


def prepareTest(path):
    # test_filenames = os.listdir(path)
    # test_df = pd.DataFrame({
    #     'filename': test_filenames
    # })
    test_df = getInput(path)
    return test_df


def convertToBinary(df):
    df = df.replace({'flower': 1, 'noFlower': 0})
    return df


def convertToString(df):
    df = df.replace({0: 'noFlower', 1: 'flower'})
    return df
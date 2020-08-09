import pandas as pd

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


def getInput(path):
    filenames = os.listdir(path)
    categories = []
    for filename in filenames:
        category = filename.split('_')[0]
        categories.append(category)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    return df


def prepareTrain(path):
    df = getInput(path)
    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    return train_df, validate_df


def prepareTest(path):
    test_df = getInput(path)
    return test_df


def convertToBinary(df):
    # le = LabelEncoder()
    # target = le.fit_transform(df)
    df = df.replace({'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4})
    return df


def convertToString(df):
    df = df.replace({0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'})
    return df


def showDivision(df):
    df['category'].value_counts().plot.bar()
    plt.show()

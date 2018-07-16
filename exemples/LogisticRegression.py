import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Karf.LinearModel import LogisticRegression
from Karf.datasets import load
from Karf.preprocessing import Scaler
from Karf.util import accuracy


def flower_2types():
    """ Exemple of how to use logistic Regression to classify two types of flowers """
    # load Data
    df = load("Iris")

    # get Species Types from dataset
    types = df["Species"].drop_duplicates().values

    # exclude one type in order to work only with two
    data = df[df.Species != types[-1]]

    # shuffle data
    data = data.iloc[np.random.permutation(len(data))]

    # map string to numbers for encoding purposes
    code_to_name = {k: types[k] for k in range(len(types))}
    name_to_code = {types[k]: k for k in range(len(types))}

    # split data to X and y and code Species names to numbers
    X = data.drop(["Id", "Species"], axis=1).astype(np.float)
    y = data["Species"].apply(lambda x: name_to_code[x])

    # split data to training sets and testing sets
    train_split = int(len(X) * 0.75)

    Xtrain = X[:train_split].values
    ytrain = y[:train_split].values
    Xtest = X[train_split:].values
    ytest = y[train_split:].values

    # normalize data
    scaler = Scaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    # initialize Logistic Regression model
    model = LogisticRegression(alpha=0.001, max_iterations=2e5)

    # fitting data
    model.fit(Xtrain, ytrain)

    # get the score
    print ("Score: ", model.score(Xtest, ytest))

    # plot learning Curves
    plt.xlabel("iteration")
    plt.ylabel("Jtrain")
    plt.plot(model.Jtrain, label="error")
    plt.legend()


def flower_3types():
    """ Exemple of how to use logistic Regression to classify 3 or more types of flowers """
    # load Data
    df = load("Iris")

    # shuffle data
    data = df.iloc[np.random.permutation(len(df))]

    # get Species types from dataset
    types = data["Species"].drop_duplicates().values

    # select input data
    X = data.drop(["Id", "Species"], axis=1).astype(np.float)

    # split input data to training and testing sets
    train_split = int(len(X) * 0.75)
    Xtrain = X[:train_split].values
    Xtest = X[train_split:].values

    # normalize data
    scaler = Scaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    results = {}
    for tp in types:
        # select ground truth data from dataset
        y = data["Species"].apply(lambda x: 1 if x == tp else 0)

        ytrain = y[:train_split].values
        # initialize the model
        model = LogisticRegression(alpha=1, max_iterations=1e4)
        # fit the model
        model.fit(Xtrain, ytrain)

        # save predicted values into a dictionary
        results[tp] = model.predict(Xtest)

    # select Species types with the heighest probability
    prediction = pd.DataFrame(results).idxmax(axis=1)

    # calculate accuracy of the model
    score = accuracy(prediction, data["Species"].values[train_split:])
    print ("Score: ", score)


if __name__ == "__main__":
    np.random.seed(1)
    # flower_2types()
    flower_3types()

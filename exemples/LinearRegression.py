import matplotlib.pyplot as plt

from Karf.LinearModel import LinearRegression
from Karf.datasets import load
from Karf.preprocessing import polynomial_variables, Scaler


def simple_function():
    """ Exemple of how to use linear Regression to fit a simple function """
    # load data
    data = load("simple_function")

    # split data into training set and testing set
    train = data.iloc[:-12].sort_values(by="X")
    test = data.iloc[-12:].sort_values(by="X")

    # split into X and y and generate polynomial variables
    Xtrain = train.drop("y", axis=1).values
    Xtrain = polynomial_variables(Xtrain, poly_degree=2)
    ytrain = train["y"].values

    Xtest = test.drop("y", axis=1).values
    ytest = test["y"].values
    Xtest = polynomial_variables(Xtest, poly_degree=2)

    # scale data
    #scaler = Scaler()
    #Xtrain = scaler.fit_transform(Xtrain)
    #Xtest = scaler.transform(Xtest)

    # initialize Linear Regression model
    model = LinearRegression(regularisation=0, max_iterations=1e5, alpha=1e-6)

    # fitting data
    model.fit(Xtrain, ytrain)

    # print the score
    print ("Score: ", model.score(Xtest, ytest))

    # save predictions from testing set
    prediction = model.predict(Xtest)

    # plot the results
    plt.figure(1)
    plt.subplot(122)

    plt.plot(Xtrain[:, 0], ytrain, label="train")
    plt.plot(Xtrain[:, 0], model.predict(Xtrain), label="predicted")
    plt.legend()
    plt.xlabel("X")

    plt.subplot(121)
    plt.plot(Xtest[:, 0], ytest, label="test")
    plt.plot(Xtest[:, 0], prediction, label="predicted")

    plt.ylabel("y")
    plt.legend()

    # plt.figure(2)
    # plt.xlabel("iteration")
    # plt.ylabel("Jtrain")
    # plt.plot(model.Jtrain,label="error")
    # plt.legend()

    plt.show()


def house_price():
    """ Exemple of how to use linear Regression to predict prices """
    # load data
    data = load("kc_house_data")

    # split into X and y
    X = data.drop(["price", "date", "id"], axis=1)
    y = data["price"]

    # split into training set and testing set
    train_split = int(len(X) * 0.75)

    Xtrain = X[:train_split].values
    ytrain = y[:train_split].values
    Xtest = X[train_split:].values
    ytest = y[train_split:].values

    # scale data
    scaler = Scaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    # initialize the model
    model = LinearRegression(alpha=1, max_iterations=20000)
    # fitting data
    model.fit(Xtrain, ytrain)

    # print the score
    print ("Score: ", model.score(Xtest, ytest))

    # plot Learning Curve
    plt.xlabel("iteration")
    plt.ylabel("Jtrain")
    plt.plot(model.Jtrain, label="error")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # house_price()
    simple_function()

from Karf.SVM import SVM
import numpy as np
import matplotlib.pyplot as plt
from Karf.datasets import load


def plot_contours(clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier """

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in """

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def linear_kernel():
    """ Solving a 2d problem using linear kernel """
    # load data
    data = load("2d_points")

    # Split data into input and output variables
    y = data["y"].values
    X = data.drop("y",axis=1).values

    y = y.astype(np.int32)
    y = y.reshape(-1, 1)

    # initialize the SVM model
    C = 1
    model = SVM(C=C, epsilon=1e-3, kernel="linear", max_iters=100)

    # fit the model
    results = model.fit(X, y)

    iteration = results['iteration']
    print "Converged after %d iterations" % iteration

    # plotting results
    w = results['w']
    b = results['b']
    xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = - (w[0] * xp + b) / w[1]
    y2 = y.reshape(1, -1)[0]

    plt.scatter(X[y2 == 1, 0], X[y2 == 1, 1], c="black")
    plt.scatter(X[y2 != 1, 0], X[y2 != 1, 1], c="red")
    plt.plot(xp, yp, '-b', c="blue", label="Decision boundery")

    X_idx = results["X"].reshape(-1, 2)
    y_idx = results["y"].reshape(1, -1)[0]
    plt.scatter(X_idx[y_idx == 1, 0], X_idx[y_idx == 1, 1], c="yellow")
    plt.scatter(X_idx[y_idx != 1, 0], X_idx[y_idx != 1, 1], c="green")

    plt.legend()
    plt.show()


def quadratic_kernel():
    """ Solving a 2d problem using quadratic kernel """
    # load data
    data = load("iris-virginica")
    # shuffle data
    data = data.iloc[np.random.permutation(len(data))]

    # split data into training and testing sets
    x_split = int(len(data) * 0.75)
    train = data[:x_split]
    test = data[x_split:]
    Xtrain = train.drop("y", axis=1).values
    Xtest = test.drop("y", axis=1).values
    ytrain = train["y"].values.reshape(-1, 1)
    ytest = test["y"].values.reshape(-1, 1)

    # initialize the model with a quadratic kernel
    model = SVM(kernel="quadratic")

    # fit the model
    results = model.fit(Xtrain, ytrain)

    iterations = results["iteration"]
    print "Converged after %d iterations" % iterations

    print "Score: %.2f"%model.score(Xtest, ytest)


if __name__ == "__main__":
    np.random.seed(2)
    #quadratic_kernel()
    linear_kernel()

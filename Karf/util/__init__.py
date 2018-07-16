import numpy as np


def cross_entropy_with_logits(labels, logits):
    """ Computes sigmoid cross entropy given logits """
    return np.mean(np.log(logits) * labels + np.log(1 - logits) * (1 - labels))


def mse(y, yhat):
    """ Computes Mean Squared Error """
    return np.mean(np.power(y - yhat, 2))


def accuracy(y, yhat):
    """ Computes Accuracy given ground truth and predicted values """
    return np.mean([x[0] == x[1] for x in zip(y, yhat)])


def regression_coef(y,yhat):
    """ Computes Regression coefficient """
    ybar = y.mean()
    return 1 - np.power(yhat - y, 2).sum() / np.power(y - ybar, 2).sum()

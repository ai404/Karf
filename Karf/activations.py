import numpy as np


def sigmoid(X):
    """ sigmoid function """
    return 1 / (1 + np.exp(-X))


def sigmoid_prime(X):
    """ sigmoid derivative """
    return sigmoid(X) * (sigmoid(X) - 1)

import numpy as np

from Karf.activations import sigmoid
from Karf.util import mse, cross_entropy_with_logits, accuracy, regression_coef


class LinearRegression(object):
    """ Linear Regression Implementation """

    def __init__(self, regularisation=0, X0=1, epsilon=1e-20, max_iterations=1e20,
                 alpha=1e-10, snapshot_steps=1000):
        self._lambda = regularisation
        self.X0 = X0
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.alpha = alpha
        self.Jtrain = []
        self.snapshot_steps = snapshot_steps

    def fit(self, X, y):
        """ fitting data using gradient decent algorithm """
        self._m = len(y)
        X.astype(np.float)
        self._features_count = X.shape[1] if len(X.shape) > 1 else 1

        self.Xtrain = np.column_stack([np.ones((self._m,)) * self.X0, X])
        self.ytrain = y

        self._coefs = np.array([1 for _ in range(self._features_count + 1)])

        print("[+] fitting in progress...")
        error = self._cost_function()
        prev_error = error + 1
        current_iter = 1
        self.Jtrain = []
        while current_iter < self.max_iterations and abs(error - prev_error) > self.epsilon:
            self.Jtrain.append(error)

            self._coefs = self._coefs - self.alpha * self._gradient()
            prev_error, error = error, self._cost_function()
            if current_iter % self.snapshot_steps == 0:
                print("\tIteration:", current_iter, "J=", error)
            current_iter += 1

        print("[-] Done.")
        print("[-] J=", error)
        print("[-] iterations=", current_iter)

    def predict(self, Xpred):
        """ make predictions """
        return self._hypothisis(Xpred)

    def score(self, Xtest, ytest):
        """ scoring results """
        predicted = self._hypothisis(Xtest)
        return regression_coef(ytest, predicted)

    def _cost_function(self):
        """ Error Calculation """
        regularisation_term = self._lambda * (self._coefs[1:] ** 2).sum() / (2 * self._m)
        return mse(self.ytrain, self._hypothisis()) / 2 + regularisation_term

    def _hypothisis(self, X=None):
        """ hypothisis function """
        X = self.Xtrain if X is None else np.column_stack([np.ones((X.shape[0],)), X])
        return X.dot(self._coefs)

    def _gradient(self):
        """ gradient function : derivative of the cost function """
        lambdas = np.ones((self._features_count + 1,)) * self._lambda
        lambdas[0] = 0

        return (self._hypothisis() - self.ytrain).dot(self.Xtrain) / self._m + lambdas * self._coefs / self._m


class LogisticRegression(LinearRegression):
    """ Logistic Regression implementation """

    def score(self, Xtest, ytest):
        predicted = [1 if x >= 0.5 else 0 for x in self._hypothisis(Xtest)]
        return accuracy(predicted, ytest)

    def _hypothisis(self, X=None):
        return sigmoid(super(LogisticRegression, self)._hypothisis(X))

    def _cost_function(self):
        regularisation_term = self._lambda * (self._coefs[1:] ** 2).sum() / (2 * self._m)
        return - cross_entropy_with_logits(self.ytrain, self._hypothisis()) + regularisation_term

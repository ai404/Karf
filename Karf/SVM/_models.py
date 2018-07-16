import numpy as np

from Karf.util import accuracy


class SVM(object):
    """ Support Vector Machine implementation """

    def __init__(self, C=1., kernel="linear", tol=1e-3, epsilon=1e-10, max_iters=1000, snapshot_steps=100):
        self.kernel = kernel
        self.C = C
        self.max_iters = max_iters
        self.tol = tol
        self.epsilon = epsilon
        self.snapshot_steps = snapshot_steps

    def fit(self, X, y):
        """ train the SVM using SMO algorithm """
        y[y == 0] = -1
        m, n = X.shape

        alphas = np.zeros((m, 1))

        E = np.zeros_like(alphas)
        self.w = np.dot(X.T, alphas * y)
        self.b = np.mean(y - np.dot(X, self.w))

        if self.kernel == "linear":
            K = self.linear_kernel(X, X)
        elif self.kernel == "gaussian":
            K = self.gaussian_kernel(X, X)
        elif self.kernel == "quadratic":
            K = self.quadratic_kernel(X, X)
        else:
            raise ValueError("%s is an unknown kernel" % self.kernel)
        iteration = 0

        while iteration < self.max_iters:
            alphas_old = np.copy(alphas)
            for i in range(m):

                E[i] = np.sign(np.dot(X[i, :], self.w) + self.b) - y[i]

                if (y[i] * E[i] < -self.tol and alphas[i, 0] < self.C) or (y[i] * E[i] > self.tol and alphas[i, 0] > 0):
                    j = int(m * np.random.rand())
                    while j == i:
                        j = int(m * np.random.rand())
                    E[j] = np.sign(np.dot(X[j, :], self.w) + self.b) - y[j]

                    alpha_j_old = alphas[j, 0]

                    if y[i] == y[j]:
                        L = max(0, alphas[j, 0] + alphas[i, 0] - self.C)
                        H = min(self.C, alphas[j, 0] + alphas[i, 0])
                    else:
                        L = max(0, alphas[j, 0] - alphas[i, 0])
                        H = min(self.C, self.C + alphas[j, 0] - alphas[i, 0])

                    if L == H:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]

                    if eta >= 0:
                        continue

                    alphas[j, 0] -= (y[j] * (E[i] - E[j])) / eta
                    alphas[j, 0] = max(L, alphas[j, 0])
                    alphas[j, 0] = min(H, alphas[j, 0])

                    if abs(alphas[j, 0] - alpha_j_old) < self.tol:
                        alphas[j, 0] = np.copy(alpha_j_old)
                        continue

                    alphas[i, 0] += y[i] * y[j] * (alpha_j_old - alphas[j, 0])

                    self.w = np.dot(X.T, alphas * y)
                    self.b = np.mean(y - np.dot(X, self.w))

            if np.linalg.norm(alphas - alphas_old) <= self.epsilon:
                break
            if (iteration + 1) % self.snapshot_steps == 0:
                print "iteration %d" % iteration
            iteration += 1

        idx = alphas > 0

        return {
            "X": X[np.where(idx), :],
            "y": y[np.where(idx)],
            "b": self.b,
            "w": self.w,
            "alphas": alphas,
            "iteration": iteration,
        }

    def predict(self, Xpred):
        """ make predictions """

        return np.sign(np.dot(Xpred, self.w) + self.b)

    def score(self, Xtest, ytest):
        """ Scoring results """

        return accuracy(self.predict(Xtest), ytest)

    def linear_kernel(self, X1, X2):
        """ Linear kernel """

        return np.dot(X1, X2.T)

    def gaussian_kernel(self, X1, X2, sigma=1):
        """ Gaussian kernel """

        raise NotImplementedError("this kernel is not implemented !")

    def quadratic_kernel(self, X1, X2):
        """ Quadratic kernel """

        return self.linear_kernel(X1, X2) ** 2

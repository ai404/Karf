import numpy as np


class Kmeans(object):
    """ Kmeans implementation """

    def __init__(self, n_clusters, max_iterations=10000, snapshot_steps=1000):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.snapshot_steps = snapshot_steps

    def fit(self, X):
        """ splitting data into n_clusters clusters """
        self._centroids = self._initialize_centroids(X)
        clusters = np.zeros(X.shape[0])
        self._distances = np.zeros((X.shape[0], self.n_clusters))

        print("[+] clustering in progress...")
        for iteration in range(self.max_iterations):

            clusters = self._find_closest_centroid(X)

            for cluster in range(self.n_clusters):
                self._centroids[cluster] = np.mean(X[clusters == cluster], axis=0)

            if iteration % self.snapshot_steps == 0:
                print("[-] iteration %d"% iteration)

        print("[-] done at iteration %d"% iteration)
        return clusters,self._centroids

    def _initialize_centroids(self, X):
        """ initialize centroids """
        return X[np.random.randint(X.shape[0], size=self.n_clusters)]

    def _find_closest_centroid(self, X):
        """ find closest centroid to each point in dataset """
        for cluster in range(self.n_clusters):
            self._distances[:, cluster] = np.linalg.norm(X - self._centroids[cluster], axis=1)

        return np.argmin(self._distances, axis=1)

    def predict(self, X):
        """ predict cluster for a given set of points """
        return self._find_closest_centroid(X)


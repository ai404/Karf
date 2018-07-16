import matplotlib.pyplot as plt

from Karf.Unsupervised import Kmeans
from Karf.datasets import make_sample_clusters

if __name__ == "__main__":
    # generate sample points with 5 clusters
    X = make_sample_clusters(n_clusters=5, n_points=1000)

    # initialize Kmeans model
    model = Kmeans(5)

    # get results from the model
    clusters,centroids = model.fit(X)

    # plot results
    plt.scatter(X[:, 0], X[:, 1], c=clusters, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="black", alpha=0.5)
    plt.show()

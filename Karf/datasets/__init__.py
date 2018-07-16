import os
import pandas as pd
import numpy as np

datasets = [
    "Iris",
    "kc_house_data",
    "simple_function",
    "iris-virginica",
    "2d_points"
]

datasets_path = os.path.dirname(__file__)


def load(name):
    """ Used to load available datasets """
    if name in datasets:

        return pd.read_csv(os.path.join(datasets_path, "%s.csv" % name))
    else:
        raise ValueError("Dataset not found!")


def make_sample_clusters(n_clusters, n_points, n_features=2, std=2, seed=1, limits=(-10, 10)):
    """ Generate square sample points """
    points_per_cluster = n_points // n_clusters
    np.random.seed(seed=seed)
    centroids = []
    for _ in range(n_features):
        centroids.append(np.random.randint(limits[0], limits[1], size=n_clusters))

    centroids = np.array(zip(*centroids))

    points = []
    for centroid in centroids:
        rands = centroid + np.random.random((points_per_cluster, n_features)) * std
        points.append(rands)

    return np.array(points).reshape(-1, 2)
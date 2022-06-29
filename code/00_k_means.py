
# %% Imports

import numpy as np


# %% Initialize k centers


def diy_kmeans_initialize(dat, k_clust):

    # Shuffle the data matrix

    dat_shuffled = np.random.shuffle(dat)

    # Choose k random points from the shuffled matrix

    k_centers = dat_shuffled[np.random.choice(
        dat_shuffled.shape[0], k_clust, replace=False), :]

    return k_centers

# %% Calculate distance matrix


def diy_kmeans_distance(dat, centers):

    d = np.empty((dat.shape[0], centers.shape[1]))

    # Find L2 norm (Euclidean distance) from each center

    for k in range(centers.shape[0]):
        d[:, k] = np.linalg.norm(dat - centers[k, :])

    # Return the squared distances

    return np.square(d)

# %%

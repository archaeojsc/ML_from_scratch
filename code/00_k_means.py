
# %% Imports

import numpy as np


# %% Initialize matrix with k centers


def diy_kmeans_initialize(dat, k_clust):

    # Shuffle the data matrix

    dat_shuffled = np.random.shuffle(dat)

    # Choose k random points from the shuffled matrix

    k_centers = dat_shuffled[np.random.choice(
        dat_shuffled.shape[0], k_clust, replace=False), :]

    return k_centers

# %% Calculate squared distance matrix S


def diy_kmeans_distance(dat, centers):

    S = np.empty((dat.shape[0], centers.shape[0]))

    # Find L2 norm (Euclidean distance) from each center

    for k in range(centers.shape[0]):
        S[:, k] = np.linalg.norm(dat - centers[k, :], axis=1)

    # Return the squared distances

    return np.square(S)

# %% Assign cluster by smallest distance


def diy_kmeans_assign(S):

    c = np.argmin(S, axis=1)

    return c


# %%

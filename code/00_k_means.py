# %% Imports

import numpy as np

# For demonstration data sets
from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt


# %% Initialize matrix with k centers


def diy_kmeans_initialize(dat, k_clust):

    # Copy the source data array

    dat_shuffled = dat.copy()

    # Shuffle the copy

    np.random.default_rng().shuffle(dat_shuffled, axis=0)

    # Choose k random points from the shuffled matrix

    k_centers = dat_shuffled[
        np.random.choice(dat_shuffled.shape[0], k_clust, replace=False), :
    ]

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

    # Label by column index of smallest value

    c = np.argmin(S, axis=1)

    return c


# %% Calculate new centers


def diy_kmeans_update_centers(dat, k_labels):

    # For each cluster label, find the mean of data points indexed by label
    # position for that cluster

    new_k_centers = np.array(
        [np.mean(dat[k_labels == k, :], axis=0) for k in np.unique(k_labels)]
    )

    return new_k_centers


# %% Stopping criteria, cluster centers have not moved


def diy_kmeans_converged(old_centers, new_centers):

    # Returns a Boolean value of equivalence

    converged = np.array_equiv(old_centers, new_centers)

    return converged


# %% DIY k-means algorithm


def diy_kmeans(dat, n_clusters, max_iter=np.inf):

    # Initial state and iteration counter
    converged = False
    i = 0

    # Initialize centers
    new_centers = diy_kmeans_initialize(dat, n_clusters)

    # Repeat until convergence or maximum iterations reached

    while (not converged) and (i <= max_iter):

        # Update iteration counter
        i += 1

        # Store copy of prior centers
        old_centers = new_centers.copy()

        # Get distances to centers as matrix S
        S = diy_kmeans_distance(dat, new_centers)

        # Assign clusters to matrix C
        C = diy_kmeans_assign(S)

        # Calculate new centers
        new_centers = diy_kmeans_update_centers(dat, C)

        # Status update for "sanity checks" every 10 iterations
        if i % 10 == 0:
            print("Iteration:\t", i)

        # Check stopping criteria
        converged = diy_kmeans_converged(old_centers, new_centers)

    # Convergence notification
    print("Converged in %s iterations." % i)

    # Create dictionary of labelled cluster centers
    centers = {k: new_centers[k, :] for k in np.unique(C)}
    labels = C

    return centers, labels


# %% Isotopic Gaussian clusters

test_blob_data, test_blob_label, test_blob_centers = make_blobs(
    n_samples=100, n_features=2, centers=3, shuffle=True, return_centers=True
)

plt.scatter(test_blob_data[:, 0], test_blob_data[:, 1], c=test_blob_label)

# %% DIY k-means on blobs

my_blob_ctr, my_blob_label = diy_kmeans(dat=test_blob_data, n_clusters=3)

plt.scatter(test_blob_data[:, 0], test_blob_data[:, 1], c=my_blob_label)


# %% Concentric circles

test_circ_data, test_circ_label = make_circles(
    n_samples=100, shuffle=True, noise=0.05, factor=0.4
)

plt.scatter(test_circ_data[:, 0], test_circ_data[:, 1], c=test_circ_label)

# %% DIY k-means on Circles

my_circ_ctr, my_circ_label = diy_kmeans(dat=test_circ_data, n_clusters=2)

plt.scatter(test_circ_data[:, 0], test_circ_data[:, 1], c=my_circ_label)

# %% "Two Moons" data set

test_moons_data, test_moons_label = make_moons(n_samples=100, shuffle=True, noise=0.1)

plt.scatter(test_moons_data[:, 0], test_moons_data[:, 1], c=test_moons_label)


# %% DIY k-means on Moons

my_moons_ctr, my_moons_label = diy_kmeans(dat=test_moons_data, n_clusters=2)

plt.scatter(test_moons_data[:, 0], test_moons_data[:, 1], c=my_moons_label)

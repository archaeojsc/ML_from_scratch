
# %% Imports

import numpy as np


# %% Initialize matrix with k centers


def diy_kmeans_initialize(dat, k_clust):

    # Copy the source data array

    dat_shuffled = dat.copy()

    # Shuffle the copy

    np.random.default_rng().shuffle(dat_shuffled, axis=0)

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

    # Label by column index of smallest value

    c = np.argmin(S, axis=1)

    return c


# %% Calculate new centers


def diy_kmeans_update_centers(dat, k_labels):

    # For each cluster label, find the mean of data points indexed by label
    # position for that cluster

    new_k_centers = np.array([np.mean(dat[k_labels == k, :], axis=0)
                             for k in np.unique(k_labels)])

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
    print('Converged in %s iterations.' % i)

    # Create dictionary of labelled cluster centers
    centers = {k: new_centers[k, :] for k in np.unique(C)}
    labels = C

    return centers, labels


# %%

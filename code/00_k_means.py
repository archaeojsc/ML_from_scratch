
def diy_kmeans_initialize(X, k_clust):

    # Shuffle the data matrix

    X_shuffled = np.random.shuffle(X)

    k_centers = X_shuffled[np.random.choice(
        X_shuffled.shape[0], k_clust, replace=False), :]

    return k_centers

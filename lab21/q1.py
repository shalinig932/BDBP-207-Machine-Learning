import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k=2, max_iters=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iters):
        labels = []

        # Assign clusters
        for point in X:
            distances = [np.linalg.norm(point - c) for c in centroids]
            labels.append(np.argmin(distances))

        labels = np.array(labels)

        # Update centroids
        new_centroids = np.array([
            X[labels == i].mean(axis=0) for i in range(k)
        ])

        # Stop if no change
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, labels


# ----------- PLOT FUNCTION -----------
def plot_clusters(X, labels, centroids):
    plt.figure()

    # Plot points
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)

    plt.title("K-Means Clustering")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()


# ----------- TEST DATA -----------
X = np.array([
    [1, 2],
    [1, 4],
    [2, 3],
    [8, 8],
    [9, 10],
    [10, 9]
])

centroids, labels = kmeans(X, k=3, max_iters=100)
plot_clusters(X, labels, centroids)

print("Centroids:\n", centroids)
print("Labels:\n", labels)

# Plot
plot_clusters(X, labels, centroids)
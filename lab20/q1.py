import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram

# -----------------------------
# Load Data
# -----------------------------
def load_data():
    from ISLP import load_data
    data = load_data("NCI60")
    X = data['data']
    y = data['labels']
    return X, y


# -----------------------------
# Scale Data
# -----------------------------
def scale_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# -----------------------------
# PCA
# -----------------------------
def perform_pca(X_scaled):
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca


# -----------------------------
# Plot Variance (Scree Plot)
# -----------------------------
def plot_variance(pca):
    var = pca.explained_variance_ratio_

    plt.figure()
    plt.plot(var, marker='o')
    plt.title("Scree Plot (Variance Explained)")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Ratio")
    plt.show()


# -----------------------------
# PCA Scatter Plot
# -----------------------------
def plot_pca(X_pca, y):
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.title("PCA - First 2 Components")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


# -----------------------------
# K-Means Clustering
# -----------------------------
def kmeans_clustering(X_scaled, k=4):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    return labels


# -----------------------------
# Plot K-Means on PCA
# -----------------------------
def plot_kmeans(X_pca, labels):
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.title("K-Means Clustering (on PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


# -----------------------------
# Hierarchical Clustering
# -----------------------------
def hierarchical_clustering(X_scaled):
    Z = linkage(X_scaled, method='complete')
    return Z


# -----------------------------
# Plot Dendrogram
# -----------------------------
def plot_dendrogram(Z):
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.show()


# -----------------------------
# MAIN
# -----------------------------
def main():
    X, y = load_data()

    X_scaled = scale_data(X)

    # PCA
    pca, X_pca = perform_pca(X_scaled)
    plot_variance(pca)
    plot_pca(X_pca, y)

    # K-Means
    k_labels = kmeans_clustering(X_scaled, k=4)
    plot_kmeans(X_pca, k_labels)

    # Hierarchical
    Z = hierarchical_clustering(X_scaled)
    plot_dendrogram(Z)


if __name__ == "__main__":
    main()
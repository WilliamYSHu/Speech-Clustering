import utils
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

def kmeans_cluster(features, n_clusters, n_rounds=5, maxrun=300):
    """
    k-means clustering
    ------
    :in:
    features: 2d array of shape (n_data, n_dim), data to cluster
    n_clusters: int, number of clusters
    n_rounds: int, number of time the k-means algorithm will be run with different centroid seeds (default:5)
    maxrun: int, maximum iterations of the k-means for a single run (default: 300)
    :out:
    labels: 1d array of shape (n_data,), index of the cluster each data point belongs to
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_rounds, max_iter=maxrun)
    kmeans.fit(features)
    labels = kmeans.predict(features)
    return labels

def hierarchical_cluster(features, n_clusters, method, metric='euclidean'):
    """
    Agglomerative (Hierarchical) clustering
    ------
    :in:
    features: 2d array of shape (n_data, n_dim), data to cluster
    method: string, methods to compute distances between clusters
            Options: "single" | "complete" | "average" | "weighted" | "centroid" | "median" | "ward"
    metric: string, distance metric to use ((default: "euclidean"))
    :out:
    Z: 2d array of shape (4, n_data-1), linkage matrix
    labels, 1d array of shape (n_data,), index of the cluster each data point belongs to
    """
    vdist = pdist(features, metric=metric)
    Z = linkage(vdist, method=method, metric=metric)
    labels = utils.linkage_to_pred(Z, n_clusters)
    return Z, labels
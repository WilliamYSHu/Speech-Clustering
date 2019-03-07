from __future__ import division
from __future__ import print_function

import json
import numpy as np
from scipy.misc import imresize
from collections import Counter, defaultdict
from sklearn.metrics.cluster import normalized_mutual_info_score


def resize(digits, row_size, column_size):
    """
    Resize images to row_size x column_size
    ------
    :in:
    digits: 3d array of shape (n_data, n_rows, n_columns)
    row_size: int, number of rows of resized image
    column_size: int, number of columns of resized image
    :out:
    digits: 3d array of shape (n_data, row_size, column_size)
    """

    return np.array([imresize(_, size=(row_size, column_size)) for _ in digits])

def get_purity(cluster_idx, class_idx):
    """
    Compute purity
    ------
    :in:
    cluster_idx: 1d array of shape (n_data), index of the cluster each data point belongs to
    class_idx: 1d array of shape (n_data), ground-truth label of each data point
    :out:
    mapping: 1d array of shape (n_cluster), mapping[i] is class label for cluster i
    cm: 2d array of shape (n_cluster, n_cluster), confusion matrix
    purity: float, purity
    """
    assert len(cluster_idx) == len(class_idx)
    n_data, n_class = len(class_idx), len(Counter(class_idx).keys())
    cm = np.zeros((n_class, n_class), dtype=np.int32)
    for i in range(n_data):
        cm[cluster_idx[i],class_idx[i]] += 1
    mapping = np.argmax(cm, axis=1)
    pred_idx, cm = np.zeros(n_data, dtype=np.int32), 0*cm
    for i in range(n_data):
        pred_idx[i] = mapping[cluster_idx[i]]
        cm[class_idx[i], pred_idx[i]] += 1
    purity = np.sum(pred_idx == class_idx)/n_data
    return mapping, cm, purity

def get_nmi(cluster_idx, class_idx):
    """
    Compute normalized mutual information
    ------
    :in:
    cluster_idx: 1d array of shape (n_data), index of the cluster each data point belongs to
    class_idx: 1d array of shape (n_data), ground-truth label of each data point
    :out:
    nmi: float, score between 0.0 and 1.0 (1.0 stands for perfectly complete labeling)
    """

    return normalized_mutual_info_score(class_idx, cluster_idx)

def gen_solution(test_pred, tune_pred, tune_labels, fname):
    """
    Generate csv file for Kaggle submission
    ------
    :in:
    test_pred: 1d array of shape (n_data), cluster index of test data
    tune_pred: 1d array of shape (n_data), cluster index of tune data
    tune_label: 1d array of shape (n_data), groundtruth label of tune data
    fname: string, name of output file
    """
    mapping, _, _ = get_purity(tune_pred, tune_labels)
    for i, _ in enumerate(test_pred):
        test_pred[i] = mapping[test_pred[i]]
    heads = ['Id', 'Category']
    with open(fname, 'w') as fo:
        fo.write(heads[0] + ',' + heads[1] + '\n')
        for i, p in enumerate(test_pred):
            fo.write(heads[0] + ' ' + str(i + 1) + ',' + str(p) + '\n')

def sample_index(ys, n_samples, n_labels=10):
    """
    Subsampling from from ys
    ------
    :in:
    ys: 1d array of shape (n_data), indices of the whole set
    n_labels: int, number of labels (default: 10)
    :out:
    indices: 1d array of shape (n_samples), sampled indices
    """
    if not (n_samples > 0 and n_samples <= len(ys)):
        raise ValueError("n_samples must be between 1 and %d" %(len(ys)))
    indices = []
    for i in range(n_labels):
        y_idx = np.arange(ys.shape[0])[(ys == i)]
        indices.append(np.random.permutation(y_idx)[:n_samples//n_labels])
    indices = np.concatenate(indices)
    return indices

def tsne_emb(features):
    """
    Run t-SNE on data
    ------
    :in:
    features: 1d array of shape (n_data, n_features), data in high-dim space
    :out:
    z_tsne: 1d array of shape (n_data, 2), embedding of data in low-dim space
    """
    print("Using TSNE")
    tsne=manifold.TSNE(perplexity=30, n_components=2, init="pca")
    z_tsne=tsne.fit_transform(np.asfarray(features, dtype="float"))
    return z_tsne


def get_label(fn, n_level):
    """
    Get labels from file, for document clustering
    ------
    :in:
    fn: string, file (.csv) with label info
    n_level: int, hierarchy level of labels
    :out:
    labels: 1d array of shape (n_data), document class labels
    titles: list of size (n_data), title of each document
    cats: list of size (n_category), list of categories
    """
    with open(fn, 'r') as fo:
        lns = [ln.strip().split(',') for ln in fo.readlines()[1:]]
    ids, titles, cats = [list(l) for l in zip(*lns)]
    cats = ["_".join(cat.split('_')[:n_level]) for cat in cats]
    for i, _ in enumerate(titles):
        titles[i] = titles[i] + " (" + cats[i] + ")"
    labels = list(cats)
    cats = list(Counter(cats).keys())
    for i, label in enumerate(labels):
        labels[i] = cats.index(label)
    return np.array(labels, dtype=np.int32), titles, cats


def linkage_to_pred(link_mat, n_clusters):
    """
    Convert linkage matrix to cluster indexes
    ------
    :in:
    link_mat: 2d array of shape (4, n_data-1), linkage matrix
    :out:
    labels, 1d array of shape (n_data,), index of the cluster each data point belongs to
    """
    n_data = link_mat.shape[0] + 1
    clusters = {}
    for i in range(n_data):
        clusters[i] = [i]
    for i in range(n_data - 1):
        z0, z1 = link_mat[i, 0], link_mat[i, 1]
        clusters[n_data+i] = clusters[z0] + clusters[z1]
        assert len(clusters[n_data+i]) == link_mat[i, 3]
        del clusters[z0], clusters[z1]
        if len(clusters) == n_clusters:
            break
    pred = np.zeros(n_data, dtype=np.int32)
    for i, idx in enumerate(clusters.keys()):
        cls = clusters[idx]
        pred[cls] = i
    return pred
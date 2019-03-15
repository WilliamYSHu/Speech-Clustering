import numpy as np
import umap
import json
import csv
import itertools
import sys

umap_dim = 20


json1_file = open('../data/pad_swbd_train_.json')
json1_str = json1_file.read()
train_data = json.loads(json1_str)

json1_file = open('../data/pad_swbd_dev_.json')
json1_str = json1_file.read()
dev_data = json.loads(json1_str)

json1_file = open('../data/pad_swbd_test_.json')
json1_str = json1_file.read()
test_data = json.loads(json1_str)

json1_data = train_data["data"] + dev_data["data"] + test_data["data"]

n_data = len(json1_data)
n_feature = len(json1_data[0])

dataMatrix = np.zeros(n_data,n_feature)

embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=umap_dim,
    random_state=42,
    ).fit_transform(dataMatrix)

np.save('umap_emb_' + str(umap_dim) + '.npy', embedding)
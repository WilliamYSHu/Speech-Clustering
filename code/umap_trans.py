import numpy as np
import umap
import json
import csv
import itertools
import sys
import umap

umap_dim = int(sys.argv[1])


json1_file = open('pad_swbd_train_.json')
json1_str = json1_file.read()
train_data = json.loads(json1_str)

json1_file = open('pad_swbd_dev_.json')
json1_str = json1_file.read()
dev_data = json.loads(json1_str)

json1_file = open('pad_swbd_test_.json')
json1_str = json1_file.read()
test_data = json.loads(json1_str)

json1_data = (train_data["data"] + dev_data["data"] + test_data["data"])[::10]

n_data = len(json1_data)
n_feature = len(json1_data[0])

dataMatrix = np.zeros((n_data,n_feature))

for idx,dataLine in enumerate(json1_data):
    dataMatrix[idx] = np.array(dataLine)

trans = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=umap_dim,
    random_state=42,
    ).fit(dataMatrix)

bundles = [(train_data["data"],'train'),(dev_data["data"],'dev'),(test_data["data"],'test')]

for dataFile,dataName in bundles:
    dataMat = np.zeros((len(dataFile), n_feature))
    for idx,dataLine in enumerate(dataFile):
        dataMat[idx] = np.array(dataLine)

    np.save('umap_emb_' + str(umap_dim) + '_' + dataName + '.npy', trans.transform(dataMat))

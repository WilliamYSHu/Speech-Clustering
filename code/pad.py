import utils
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cosine, euclidean
from numpy.random import permutation as rpm
from plot import plot_images, plot_confusion_matrix
from utils import resize, sample_index, gen_solution, get_purity, get_nmi
from numpy.linalg import norm

import json
import csv
import itertools
import sys

# load raw signal data
data=np.load('../data/swbd.npy').item()
def stack_frames(num_frames,timeseries):
    pad = num_frames//2
    pad_timeseries = np.pad(timeseries,((pad,pad),(0,0)),"constant")
    res = np.zeros((timeseries.shape[0],num_frames*timeseries.shape[1]))
    for i in range(pad,timeseries.shape[0]+pad):
        temp= []
        for j in range(i-pad,i+pad+1):
            temp.append(pad_timeseries[j])
        res[i-pad] = np.concatenate(temp)
    return res

#print(stack_frames(3,np.array([[1,2,3],[1,2,3]])))
for c in ["train","dev","test"]:
    print(1)
    t_data = data[c]["data"]
    pad_t_data = list(map(lambda x: stack_frames(5,x), t_data))
    pad_frames = []
    
    for i in pad_t_data: 
        for j in range(i.shape[0]):
            pad_frames.append(i[j].tolist())
    print(2)
    
    frame_labels = []
    for i in range(len(pad_t_data)):
        for j in range(pad_t_data[i].shape[0]):
            frame_labels.append((i,j))

    check_num_frames = 0
    for i in range(len(t_data)):
        check_num_frames += t_data[i].shape[0]
    print("check frame numbers " + str(check_num_frames) + " vs " + str(len(frame_labels)) + " vs " +  str(len(pad_frames)))
    temp = {}
    temp["data"] = pad_frames
    temp["frames"] = frame_labels
    temp["labels"] =data[c]["labels"] 
    with open('pad_swbd_' + c + '_.json', 'w') as fp:
        json.dump(temp, fp)

json_data =  data 

''' 
spark = SparkSession\
        .builder\
        .master("spark://yiyangou-VirtualBox:7077")\
        .appName("DTW")\
        .config("spark.cores.max", "6")\
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")
numPartitions=6

for c in ["train","dev","test"]:
    print(1)
    for i in range(len(json_data[c]["data"])):
        json_data[c]["data"][i] = json_data[c]["data"][i].tolist()
'''

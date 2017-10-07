# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 23:55:50 2017

@author: Stefan Draghici
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

blobs=make_blobs(n_samples=500, n_features=5, centers=8, cluster_std=1.8, random_state=101)

plt.scatter(blobs[0][:, 0], blobs[0][:, 1])

kmeans=KMeans(n_clusters=8)
kmeans.fit(blobs[0])
kmeans.cluster_centers_
kmeans.labels_

fig, (ax1,ax2)=plt.subplots(1, 2, sharey=True)

ax1.set_title('Kmeans')
ax1.scatter(blobs[0][:,0], blobs[0][:,1], c=kmeans.labels_, cmap='rainbow')
ax2.set_title('Original')
ax2.scatter(blobs[0][:,0], blobs[0][:,1], c=blobs[1], cmap='rainbow')
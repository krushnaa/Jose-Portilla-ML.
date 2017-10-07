# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 23:55:50 2017

@author: Stefan Draghici
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('College_Data', index_col=0)
sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, hue='Private', fit_reg=False)
sns.lmplot(x='Outstate', y='F.Undergrad', data=df, hue='Private', fit_reg=False)

df[df['Grad.Rate']>100]
df['Grad.Rate']['Cazenovia College']=100

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2)
kmeans.fit(df.drop('Private', axis=1))
kmeans.cluster_centers_

def converter(private):
    if private=='yes':
        return 1
    else:
        return 0
    
df['Cluster']=df['Private'].apply(converter)

from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(df['Cluster'], kmeans.labels_)
cr=classification_report(df['Cluster'], kmeans.labels_)
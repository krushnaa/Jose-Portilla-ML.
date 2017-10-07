# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 11:45:09 2017

@author: Stefan Draghici
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

can=load_breast_cancer()
df=pd.DataFrame(can['data'], columns=can['feature_names'])

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaled_df=scaler.fit_transform(df)

from sklearn.decomposition import PCA

pca=PCA(n_components=2)
pca.fit(scaled_df)
x_pca=pca.transform(scaled_df)


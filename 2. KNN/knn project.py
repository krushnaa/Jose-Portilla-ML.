#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:33:36 2017

@author: user
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('KNN_Project_Data', index_col=0)
sns.pairplot(df, hue='TARGET CLASS', pallette='coolwarm')

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features=scaler.transform(df.drop('TARGET CLASS', axis=1))
df_features=pd.DataFrame(scaled_features, columns=df.columns[:-1])

from sklearn.model_selection import train_test_split

X=df_features
y=df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
predictions=knn.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(y_test, predictions)
cr=classification_report(y_test, predictions)

error_rate=[]
for i in range(1, 40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))
    
plt.plot(range(1, 40), error_rate)

knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
predictions=knn.predict(X_test)

cm=confusion_matrix(y_test, predictions)
cr=classification_report(y_test, predictions)
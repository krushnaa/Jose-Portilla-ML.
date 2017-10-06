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

df=pd.read_csv('kyphosis.csv')

sns.pairplot(df, hue='Kyphosis')

X=df.drop('Kyphosis', axis=1)
y=df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()
tree.fit(X_train, y_train)
predictions=tree.predict(X_test)

cr=classification_report(y_test, predictions)
cm=confusion_matrix(y_test, predictions)

from sklearn.ensemble import RandomForestClassifier
forrest=RandomForestClassifier(n_estimators=1)
forrest.fit(X_train, y_train)
forrest_predictions=forrest.predict(X_test)

cr_forrest=classification_report(y_test, forrest_predictions)
cm_forrest=confusion_matrix(y_test, forrest_predictions)

error_rate=[]
for i in range(1, 101):
    rfc=RandomForestClassifier(n_estimators=i)
    rfc.fit(X_train, y_train)
    pred_i=rfc.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))
    
plt.plot(range(1, 101), error_rate)

forrest=RandomForestClassifier(n_estimators=200)
forrest.fit(X_train, y_train)
forrest_predictions=forrest.predict(X_test)

cr_forrest=classification_report(y_test, forrest_predictions)
cm_forrest=confusion_matrix(y_test, forrest_predictions)
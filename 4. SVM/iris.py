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

iris=sns.load_dataset('iris')
setosa=iris[iris['species']=='setosa']

df_feat=pd.DataFrame(c_data['data'], columns=c_data['feature_names'])

X=iris.drop('species', axis=1)
y=iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.svm import SVC
svm=SVC()
svm.fit(X_train, y_train)
predictions=svm.predict(X_test)

cr=classification_report(y_test, predictions)
cm=confusion_matrix(y_test, predictions)
 
# Use grid search to find the best params to tune
from sklearn.model_selection import GridSearchCV

param_grid={'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001]}
grid=GridSearchCV(estimator=svm, param_grid=param_grid, verbose=3)
grid.fit(X_train, y_train)
grid.best_params_

grid_preds=grid.predict(X_test)

cr_grid=classification_report(y_test, grid_preds)
cm_grid=confusion_matrix(y_test, grid_preds)
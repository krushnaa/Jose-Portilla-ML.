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

loans=pd.read_csv('loan_data.csv')

loans[loans['credit.policy']==1]['fico'].hist(bins=35, color='blue', label='credit policy = 1', alpha=0.5)
loans[loans['credit.policy']==0]['fico'].hist(bins=35, color='red', label='credit policy = 0', alpha=0.5)
plt.legend()

cat_feats=['purpose']
final_data=pd.get_dummies(loans, columns=cat_feats, drop_first=True)

X=final_data.drop('not.fully.paid', axis=1)
y=final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()
tree.fit(X_train, y_train)
predictions=tree.predict(X_test)

cr_tree=classification_report(y_test, predictions)
cm_tree=confusion_matrix(y_test, predictions)
 
from sklearn.ensemble import RandomForestClassifier
forrest=RandomForestClassifier(n_estimators=50)
forrest.fit(X_train, y_train)
forrest_predictions=forrest.predict(X_test)

cr_forrest=classification_report(y_test, forrest_predictions)
cm_forrest=confusion_matrix(y_test, forrest_predictions)
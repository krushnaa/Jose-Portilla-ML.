# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 23:55:50 2017

@author: Stefan Draghici
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

ad_data=pd.read_csv('advertising.csv')

ad_data['Age'].plot.hist(bins=40)
sns.jointplot(x='Age', y='Area Income', data=ad_data)
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, kind='kde')
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data)

X=ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y=ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

regressor=LogisticRegression()
regressor.fit(X_train, y_train)
predictions=regressor.predict(X_test)
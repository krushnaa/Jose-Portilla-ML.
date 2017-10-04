# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 22:27:49 2017

@author: Stefan Draghici
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset=pd.read_csv('Ecommerce Customers')

sns.jointplot(dataset['Time on Website'], dataset['Yearly Amount Spent'])
sns.jointplot(dataset['Time on App'], dataset['Length of Membership'], kind='hex')
sns.pairplot(dataset)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=dataset)

X=dataset[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

y=dataset[['Yearly Amount Spent']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor=LinearRegression()
regressor.fit(X_train, y_train)

predictions=regressor.predict(X_test)

plt.scatter(y_test, predictions)
sns.distplot(y_test-predictions)

mae=metrics.mean_absolute_error(y_test, predictions)
mse=metrics.mean_squared_error(y_test, predictions)
mre=np.sqrt(metrics.mean_squared_error(y_test, predictions))
variance=metrics.explained_variance_score(y_test, predictions)

cdf=pd.DataFrame(regressor.coef_, columns=X.columns)
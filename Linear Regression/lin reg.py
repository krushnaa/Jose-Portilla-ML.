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

dataset=pd.read_csv('USA_Housing.csv')
#sns.pairplot(dataset)
#sns.distplot(dataset['Price'])
#sns.heatmap(dataset.corr())

X=dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
           'Avg. Area Number of Bedrooms', 'Area Population']]

y=dataset[['Price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor=LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.intercept_)
print(regressor.coef_)

coef_data_frame=pd.DataFrame(data=regressor.coef_, columns=X.columns)

predictions=regressor.predict(X_test)

plt.scatter(y_test, predictions)
sns.distplot(y_test-predictions)

mae=metrics.mean_absolute_error(y_test, predictions)
mse=metrics.mean_squared_error(y_test, predictions)
mre=np.sqrt(metrics.mean_squared_error(y_test, predictions))
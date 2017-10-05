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

train=pd.read_csv('titanic_train.csv')

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.countplot(x='Survived', hue='Sex', data=train)
sns.distplot(train['Age'].dropna(), kde=False, bins=30)
sns.countplot(x='SibSp', data=train)
sns.boxplot(x='Pclass', y='Age', data=train)
train['Fare'].iplot(kind='hist', bins=30)

def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
    
train['Age']=train[['Age', 'Pclass']].apply(impute_age, axis=1)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)

train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)

sex=pd.get_dummies(train['Sex'], drop_first=True)
embark=pd.get_dummies(train['Embarked'], drop_first=True)
train=pd.concat([train, sex, embark], axis=1)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train.drop('PassengerId', axis=1, inplace=True)

X=train.drop('Survived', axis=1)
y=train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

regressor=LogisticRegression()
regressor.fit(X_train, y_train)

predictions=regressor.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
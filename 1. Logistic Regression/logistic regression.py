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

train=pd.read_csv('titanic_train.csv')

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.countplot(x='Survived', hue='Sex', data=train)
sns.distplot(train['Age'].dropna(), kde=False, bins=30)
sns.countplot(x='SibSp', data=train)

train['Fare'].iplot(kind='hist', bins=30)
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 21:05:58 2017

@author: Stefan Draghici
"""

import pandas as pd
import numpy as np

yelp=pd.read_csv('yelp.csv')
yelp['text length']=yelp['text'].apply(len)

import seaborn as sns
import matplotlib.pyplot as plt

facet_grid=sns.FacetGrid(yelp, col='stars')
facet_grid.map(plt.hist, 'text length', bins=50)

stars=yelp.groupby('stars').mean()
one_five_reviews=yelp[(yelp['stars']==1) | (yelp['stars']==5)]

X=one_five_reviews['text']
y=one_five_reviews['stars']

from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(X_train, y_train)
predictions=nb.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(y_test, predictions)
cr=classification_report(y_test, predictions)

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
pipeline=Pipeline([('bow', CountVectorizer()), ('tfidf', TfidfTransformer()), ('classifier', MultinomialNB())])
X=one_five_reviews['text']
y=one_five_reviews['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pipeline.fit(X_train, y_train)
pipeline_predictions=pipeline.predict(X_test)
cm=confusion_matrix(y_test, pipeline_predictions)
cr=classification_report(y_test, pipeline_predictions)
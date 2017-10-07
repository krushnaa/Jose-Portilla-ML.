# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:56:59 2017

@author: Stefan Draghici
"""

import numpy as np
import pandas as pd

columns_names=['user_id', 'item_id', 'rating', 'timestamp']

df=pd.read_csv('u.data', sep='\t', names=columns_names)

movie_titles=pd.read_csv('Movie_Id_Titles')
df=pd.merge(df, movie_titles, on='item_id')

import matplotlib.pyplot as plt
import seaborn as sns

df.groupby('title')['rating'].mean().sort_values(ascending=False)
df.groupby('title')['rating'].count().sort_values(ascending=False)

ratings=pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings']=pd.DataFrame(df.groupby('title')['rating'].count())
ratings['num of ratings'].hist(bins=100)
moviemat=df.pivot_table(index='user_id', columns='title', values='rating')
ratings.sort_values('num of ratings', ascending=False)

starwars_user_ratings=moviemat['Star Wars (1977)']
liarliar_user_ratings=moviemat['Liar Liar (1997)']
similar_to_starwars=moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar=moviemat.corrwith(liarliar_user_ratings)

corr_starwars=pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
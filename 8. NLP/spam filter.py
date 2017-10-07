# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 19:07:24 2017

@author: Stefan Draghici
"""

messages=[line.rstrip() for line in open('SMSSpamCollection')]

for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    
import pandas as pd

messages=pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
messages.groupby('label').describe()
messages['length']=messages['message'].apply(len)

import matplotlib.pyplot as plt
import seaborn as sns

messages['length'].plot.hist(bins=50)
messages[messages['length']==910]

from nltk.corpus import stopwords
import string

no_punctuation=[c for c in message if c not in string.punctuation]
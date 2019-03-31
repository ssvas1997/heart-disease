# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:13:54 2019

@author: Srinivas
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

df= pd.read_csv("E:\AI project\heart.csv")
df.dtypes
features = df.drop('target',axis=1).values
classes = df['target'].values

(train_feat,test_feat,train_class,test_class) = train_test_split(features,classes,train_size = 0.7,random_state=42)

knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(train_feat,train_class)
pred = knn.predict(test_feat)

print("accuracy",metrics.accuracy_score(test_class,pred))

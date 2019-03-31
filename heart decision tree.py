# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

df= pd.read_csv("E:\AI project\heart.csv")
df.dtypes
df.describe()
features=df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
classes=df['target'].values
features.shape
(train_feat,test_feat,train_classes,test_classes)= train_test_split(features,classes,train_size=0.7,random_state=1)
#Training
dectree= DecisionTreeClassifier()
dectree.fit(train_feat,train_classes) #Supervised Learning
#Testing
pred= dectree.predict(test_feat)
print("Accuracy:",metrics.accuracy_score(test_classes,pred))
#Predicting a single input feature

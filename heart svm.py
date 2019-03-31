# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:30:25 2019

@author: Srinivas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df= pd.read_csv("E:\AI project\heart.csv")
x = df.drop('target',axis=1).values
y = df['target'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

from sklearn.svm import SVC
svclassifier =SVC(kernel='polynomial')
svclassifier.fit(x_train,y_train)
y_pred=svclassifier.predict(x_test)

print("accuracy",metrics.accuracy_score(y_test,y_pred))
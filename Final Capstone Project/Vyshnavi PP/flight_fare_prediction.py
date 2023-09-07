# -*- coding: utf-8 -*-
"""Flight fare Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r1WBIYWf8MxwfC7qjv7RaIFSvCyTIE-6
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import random

Train_data = pd.read_csv("/content/Data_Train.csv")

Train_data.head()

Train_data.info()

#find out how many loan cases are Paid Off, Collection or Collection_PaidOff status
plt.figure(figsize=(8, 6))
sns.countplot(x="Source", data=Train_data)
plt.xlabel('Source')
plt.ylabel('Price')
plt.title('location and price graph')
plt.xticks(rotation=45)
plt.show()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for column in Train_data.columns:
  if Train_data[column].dtype == "object":
    Train_data[column] = le.fit_transform(Train_data[column])

Train_data.info()

Train_data.describe()

Train_data.isna().sum()

Train_data.shape

Train_data.head()

plt.figure(figsize = (18,18))
sns.heatmap(Train_data.corr(), annot = True, cmap = "RdYlGn")

plt.show()

x = Train_data.drop('Price',axis=1)
y = Train_data['Price']

x.ndim

y.shape

from sklearn.model_selection import train_test_split

xtrain,xtest, ytrain,ytest = train_test_split(x,y,test_size=.25)

from sklearn.svm import SVR
svr = SVR().fit(xtrain,ytrain)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor().fit(xtrain,ytrain)

model_names = ["LR","Lasso","Ridge","SVR","RandomForest"]
accuracy = [lr.score(xtest,ytest),lasso.score(xtest,ytest),ridge.score(xtest,ytest),svr.score(xtest,ytest),rfr.score(xtest,ytest)]

import matplotlib.pyplot as plt

plt.bar(model_names, accuracy)

for i in range(len(model_names)):
  plt.text(i,accuracy[i],str(round(accuracy[i],2)),ha="center", bbox=dict(facecolor="wheat",alpha=0.5))

rfr.score(x,y)

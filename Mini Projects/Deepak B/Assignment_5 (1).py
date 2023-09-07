#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import seaborn as sns


# In[ ]:


sns.get_dataset_names()


# In[ ]:


data = sns.load_dataset("car_crashes")


# In[ ]:


data.columns


# In[ ]:


data.info()


# In[ ]:


data.isna().sum()


# In[ ]:


data.dropna(inplace = True)


# In[ ]:


data.describe()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for column in data.columns:
  if data[column].dtype == "object":
    data[column] = le.fit_transform(data[column])


# In[ ]:


data.info()


# In[ ]:


data = data[data.columns[2:]]


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


x, y = data.drop("not_distracted",axis = 1), data["not_distracted"]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.25)


# In[ ]:


lr = LinearRegression().fit(xtrain, ytrain)
lasso = Lasso().fit(xtrain, ytrain)
ridge = Ridge().fit(xtrain, ytrain)


# In[ ]:


lr.score(xtrain, ytrain)


# In[ ]:


lasso.score(xtrain, ytrain)


# In[ ]:


ridge.score(xtrain, ytrain)


# In[ ]:


lr.score(xtrain, ytrain) > lasso.score(xtrain, ytrain)


# In[ ]:


lr.score(xtrain, ytrain) > ridge.score(xtrain, ytrain)


# In[ ]:


lasso.score(xtrain, ytrain) > ridge.score(xtrain, ytrain)


# In[ ]:


lasso.score(xtrain, ytrain) < ridge.score(xtrain, ytrain)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[21]:


data = pd.read_csv(r"C:\Users\durga\OneDrive\Desktop\New folder\Impaired_Driving_Death_Rate__by_Age_and_Gender__2012___2014__All_States.csv",encoding='iso-8859-1')


# In[27]:


data.head()

data.info()

data.isna().sum()

data.dropna(inplace = True)

data.describe()

data.isna().sum()


# In[37]:


data= data[["Male, 2012","Location"]]

data.info()


# In[32]:


from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder


# In[34]:


le = LabelEncoder()
for column in data.columns:
  if data[column].dtype == "object":
    data[column] = le.fit_transform(data[column])

data.info()

data.head()

from pandas.core.internals.blocks import NumericBlock
inertia = []
for number_of_cluster in range(1,5):
  kmean = KMeans(n_clusters = number_of_cluster)
  kmean.fit(data)
  inertia.append(kmean.inertia_)

plt.plot(range(1,5),inertia,"*r-")

kmean = KMeans(n_clusters=2)
predict = kmean.fit_predict(data)

predict

x = data

plt.scatter(x.iloc[predict == 0,0],x.iloc[predict == 0,0],c="r")
plt.scatter(x.iloc[predict == 1,0],x.iloc[predict == 1,0],c="b")

plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],s = 100, c="black")

plt.title("Male, 2012 vs Location")
plt.ylabel("Male, 2012")
plt.xlabel("Location")


# In[38]:


le = LabelEncoder()
for column in data.columns:
  if data[column].dtype == "object":
    data[column] = le.fit_transform(data[column])

data.info()

data.head()

from pandas.core.internals.blocks import NumericBlock
inertia = []
for number_of_cluster in range(1,5):
  kmean = KMeans(n_clusters = number_of_cluster)
  kmean.fit(data)
  inertia.append(kmean.inertia_)

plt.plot(range(1,5),inertia,"*r-")

kmean = KMeans(n_clusters=2)
predict = kmean.fit_predict(data)

predict

x = data

plt.scatter(x.iloc[predict == 0,0],x.iloc[predict == 0,0],c="r")
plt.scatter(x.iloc[predict == 1,0],x.iloc[predict == 1,0],c="b")

plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],s = 100, c="black")

plt.title("Male, 2014 vs Location")
plt.ylabel("Male, 2014")
plt.xlabel("Location")


# In[39]:


le = LabelEncoder()
for column in data.columns:
  if data[column].dtype == "object":
    data[column] = le.fit_transform(data[column])

data.info()

data.head()

from pandas.core.internals.blocks import NumericBlock
inertia = []
for number_of_cluster in range(1,5):
  kmean = KMeans(n_clusters = number_of_cluster)
  kmean.fit(data)
  inertia.append(kmean.inertia_)

plt.plot(range(1,5),inertia,"*r-")

kmean = KMeans(n_clusters=2)
predict = kmean.fit_predict(data)

predict

x = data

plt.scatter(x.iloc[predict == 0,0],x.iloc[predict == 0,0],c="r")
plt.scatter(x.iloc[predict == 1,0],x.iloc[predict == 1,0],c="b")

plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],s = 100, c="black")

plt.title("Female, 2012 vs Location")
plt.ylabel("Female, 2012")
plt.xlabel("Location")


# In[40]:


le = LabelEncoder()
for column in data.columns:
  if data[column].dtype == "object":
    data[column] = le.fit_transform(data[column])

data.info()

data.head()

from pandas.core.internals.blocks import NumericBlock
inertia = []
for number_of_cluster in range(1,5):
  kmean = KMeans(n_clusters = number_of_cluster)
  kmean.fit(data)
  inertia.append(kmean.inertia_)

plt.plot(range(1,5),inertia,"*r-")

kmean = KMeans(n_clusters=2)
predict = kmean.fit_predict(data)

predict

x = data

plt.scatter(x.iloc[predict == 0,0],x.iloc[predict == 0,0],c="r")
plt.scatter(x.iloc[predict == 1,0],x.iloc[predict == 1,0],c="b")

plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],s = 100, c="black")

plt.title("Female, 2014 vs Location")
plt.ylabel("Female, 2014")
plt.xlabel("Location")


# In[42]:


le = LabelEncoder()
for column in data.columns:
  if data[column].dtype == "object":
    data[column] = le.fit_transform(data[column])

data.info()

data.head()

from pandas.core.internals.blocks import NumericBlock
inertia = []
for number_of_cluster in range(1,5):
  kmean = KMeans(n_clusters = number_of_cluster)
  kmean.fit(data)
  inertia.append(kmean.inertia_)

plt.plot(range(1,5),inertia,"*r-")

kmean = KMeans(n_clusters=2)
predict = kmean.fit_predict(data)

predict

x = data

plt.scatter(x.iloc[predict == 0,0],x.iloc[predict == 0,0],c="r")
plt.scatter(x.iloc[predict == 1,0],x.iloc[predict == 1,0],c="b")

plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],s = 100, c="black")

plt.title("All Ages, 2012 vs Location")
plt.ylabel("All Ages, 2012")
plt.xlabel("Location")


# In[43]:


le = LabelEncoder()
for column in data.columns:
  if data[column].dtype == "object":
    data[column] = le.fit_transform(data[column])

data.info()

data.head()

from pandas.core.internals.blocks import NumericBlock
inertia = []
for number_of_cluster in range(1,5):
  kmean = KMeans(n_clusters = number_of_cluster)
  kmean.fit(data)
  inertia.append(kmean.inertia_)

plt.plot(range(1,5),inertia,"*r-")

kmean = KMeans(n_clusters=2)
predict = kmean.fit_predict(data)

predict

x = data

plt.scatter(x.iloc[predict == 0,0],x.iloc[predict == 0,0],c="r")
plt.scatter(x.iloc[predict == 1,0],x.iloc[predict == 1,0],c="b")

plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],s = 100, c="black")

plt.title("All Ages, 2014 vs Location")
plt.ylabel("All Ages, 2014")
plt.xlabel("Location")


# In[ ]:





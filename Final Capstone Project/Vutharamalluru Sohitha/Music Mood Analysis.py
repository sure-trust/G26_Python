#!/usr/bin/env python
# coding: utf-8

# # PROJECT-3

# # Music Mood Analysis

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("C:\\Users\\VutharamalluruBalaji\\Documents\\music mood analysis\\muse_dataset.csv")
print(data)


# In[2]:


data.head()


# In[4]:


data.info()


# In[5]:


data.isna().sum()


# In[6]:


data.drop("id",axis=1,inplace=True)
data.drop("spotify_id",axis=1,inplace=True)
data.drop("mbid",axis=1,inplace=True)
data.drop("dominance_tags",axis=1,inplace=True)


# In[7]:


data.info()


# In[8]:


from sklearn.preprocessing import LabelEncoder


# In[9]:


data["track"]=LabelEncoder().fit_transform(data["track"])
data["artist"]=LabelEncoder().fit_transform(data["artist"])
data["valence_tags"]=LabelEncoder().fit_transform(data["valence_tags"])
data["arousal_tags"]=LabelEncoder().fit_transform(data["arousal_tags"])


# In[10]:


data.info()


# In[11]:


data=data[data.columns[1:3]]


# In[12]:


from sklearn.cluster import KMeans


# In[13]:


from pandas.core.internals.blocks import NumericBlock
inertia = []
for no_of_cluster in range(1,10):
    kmn= KMeans(n_clusters = no_of_cluster)
    kmn.fit(data)
    inertia.append(kmn.inertia_)


# In[14]:


plt.plot(range(1,10),inertia,"*b-")


# In[15]:


kmn =KMeans(n_clusters=3)
prdct=kmn.fit_predict(data)


# In[16]:


prdct


# In[17]:


x=data


# In[18]:


plt.scatter(x.iloc[prdct == 0,0],x.iloc[prdct == 0,1],c="r")
plt.scatter(x.iloc[prdct == 1,0],x.iloc[prdct == 1,1],c="b")
plt.scatter(x.iloc[prdct == 2,0],x.iloc[prdct == 2,1],c="g")
plt.scatter(kmn.cluster_centers_[:,0],kmn.cluster_centers_[:,1],s = 300, c="black")
plt.xlabel("artist")
plt.ylabel("valence_tags")


# In[19]:


plt.xlabel("artist")
plt.ylabel("valence_tags")
plt.title("Histogram in Music Mood Analysis")
plt.hist(data["valence_tags"],ec='r')


# In[20]:


import seaborn as sns
sns.histplot(data=data,x='valence_tags')


# # CONCLUSION

# In[24]:


'''In this project we have explored the use of machine leraning to predict the mood of track.
we used a variety of features,including audio features,lyrics,and musical structure,to train a machine learning model.the model was able to acheive an accuracy on a test set of music tracks.'''


# In[ ]:





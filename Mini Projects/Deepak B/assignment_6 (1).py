#!/usr/bin/env python
# coding: utf-8

# # K-Means clustering

# In[84]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[85]:


data = pd.read_csv('student_Marks.csv')


# In[86]:


X = data[['number_courses', 'time_study', 'Marks']]


# In[87]:


inertia_values = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)


# In[88]:


plt.plot(range(1, 10), inertia_values, marker='*')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


# In[89]:


num_clusters =3


# In[90]:


kmeans = KMeans(n_clusters=num_clusters, random_state=0)
data['kmeans_cluster'] = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_


# In[65]:


plt.scatter(data['time_study'], data['Marks'], c=data['kmeans_cluster'], cmap='rainbow')
plt.scatter(centroids[:,1], centroids[:,2], c='black', marker='*', s=200, label='Centroids')
plt.xlabel('Study Time')
plt.ylabel('Marks')
plt.title('K-Means Clustering')
plt.show()


# In[53]:


'''cluster 1 students study 6 to 8 hours and scored 40 and above, less attention is needed
cluster 2 students study 3 to 6 hours and scored 20 to 40 marks little bit more  attention is needed for then compare to cluster 1 students
cluster 3 students study 0 to 3 hours and scored less than 20 marks ,so more attention is needed comapare to the other cluster '''


# In[55]:


plt.scatter(data['number_courses'], data['Marks'], c=data['kmeans_cluster'], cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 2], c='green', marker='o', s=200, label='Centroids')
plt.xlabel('Number of Cources taken by students')
plt.ylabel('Marks Scored')
plt.title('K-Means Clustering')
plt.show()


# In[56]:


''' In above graph indicates that students who took 8 cources and scored 40 and above they are good in maneging time 
and need less attention and students who scored less marks although they took 3 scores need more attention (i.e students who are marked in purple need more attention) '''


# In[58]:


plt.scatter(data['number_courses'], data['time_study'], c=data['kmeans_cluster'], cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 1], c='green', marker='o', s=200, label='Centroids')
plt.xlabel('Number of Cources taken by students')
plt.ylabel('Study Time')
plt.title('K-Means Clustering')
plt.show()


# In[32]:


'''In the above graph shows that some students study less than 2-3 hours even they took 7-8 number of cources need more attention compare to students who study more than 5 hours '''


# # Hierarchical clustering

# In[91]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[92]:


data = pd.read_csv("student_Marks.csv")


# In[93]:


data.head()


# In[94]:


data = data[data.columns[-2:]]


# In[95]:


data.head()


# In[96]:


data.isna().sum()


# In[98]:


from scipy.cluster import hierarchy

d = hierarchy.dendrogram(hierarchy.linkage(data))
plt.xlabel('Data Points')
plt.ylabel('distance between the clusters')
plt.title('Hierarchical clustering')
plt.show()


# In[99]:


''' In the above Hierarchical clustering the y axis represnts the distance between the clusters and x axis represents the  data points'''


# # AgglomerativeClustering

# In[101]:


from sklearn.cluster import AgglomerativeClustering

cl = AgglomerativeClustering(n_clusters = 4)
pred = cl.fit_predict(data)


# In[102]:


pred


# In[106]:


plt.scatter(data.iloc[ pred == 0, 0],data.iloc[ pred == 0, 1], c= "red")
plt.scatter(data.iloc[ pred == 1, 0],data.iloc[ pred == 1, 1], c= "blue")
plt.scatter(data.iloc[ pred == 2, 0],data.iloc[ pred == 2, 1], c= "green")
plt.scatter(data.iloc[ pred == 3, 0],data.iloc[ pred == 3, 1], c= "yellow")


# In[ ]:





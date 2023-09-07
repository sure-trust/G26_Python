#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# In[35]:


data=pd.read_csv("BangaloreZomatoData.csv",na_values=['-'])


# In[36]:


data1=data.copy()


# In[37]:


data1


# In[38]:


data1.info()


# In[39]:


data1.isnull().sum()


# # Data Cleaning 

# In[40]:


data1.drop(['KnownFor','PopularDishes','PeopleKnownFor','Timing'],axis=1,inplace=True)


# In[41]:


data1['Delivery Ratings'].unique()


# In[42]:


data1.describe()


# In[43]:


data1['Dinner Ratings'].mean()


# In[44]:


data1['Dinner Ratings'].fillna(data1['Dinner Ratings'].mean(),inplace=True)


# In[45]:


data1['Delivery Ratings'].mean()


# In[46]:


data1['Delivery Ratings'].fillna(data1['Delivery Ratings'].mean(),inplace=True)


# In[47]:


data1


# In[48]:


data1.isnull().sum()


# # Data Visualization

# In[49]:


data1['IsHomeDelivery'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Proportion of Restaurants Offering Home Delivery')
plt.ylabel('')
plt.show()


# In[50]:


data1['IsHomeDelivery'] = data1['IsHomeDelivery'].replace({'Not Available':0, 'Available':1})
counts = data1['IsHomeDelivery'].value_counts()
print(counts)


# In[51]:


data1['isTakeaway'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Proportion of Restaurants Offering Takeaway to the customers')
plt.ylabel('')
plt.show()


# In[52]:


data1['isTakeaway'] = data1['isTakeaway'].replace({'Not Available':0,'Available':1})
counts = data1['isTakeaway'].value_counts()
print(counts)


# In[53]:


data1['AverageCost'].plot(kind='hist', bins=20)
plt.xlabel('Average Cost')
plt.ylabel('Frequency')
plt.title('Distribution of Average Costs')
plt.show()


# In[54]:


data1['isVegOnly'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Proportion of Restaurants Offering only Vegetarian food to the customers')
plt.ylabel('')
plt.show()


# In[55]:


data1['isVegOnly'] = data1['isVegOnly']
counts = data1['isVegOnly'].value_counts()
print(counts)


# In[56]:


data1['Dinner Ratings'] = pd.to_numeric(data1['Dinner Ratings'], errors='coerce')


# In[57]:


filtered_ratings = data1['Dinner Ratings'].dropna()
plt.figure(figsize=(8, 6))  
plt.hist(filtered_ratings, bins=[1, 2, 3, 4, 5, 6], edgecolor='black', alpha=0.7)
plt.xlabel('Dinner Ratings')
plt.ylabel('Count')
plt.title('Distribution of Dinner Ratings')
plt.xticks([1, 2, 3, 4, 5])
plt.show()


# In[58]:


data1['Dinner Ratings'] = pd.to_numeric(data1['Dinner Ratings'], errors='coerce')
filtered_ratings = data1['Dinner Ratings'].dropna()
rating_counts = filtered_ratings.value_counts().sort_index()
print("Dinner Ratings Counts:")
print("Dinner Ratings Counts:")
for rating, count in rating_counts.items():
    print(f"Rating {rating}: {count}")


# In[59]:


numerical_columns = data1.select_dtypes(include=['number'])
corr_matrix = numerical_columns.corr()
plt.figure(figsize=(12, 8))
sns.set(font_scale=1.2)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[60]:


data1


# # K-Means

# In[64]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
selected_columns = ['Dinner Ratings', 'Delivery Ratings', 'isVegOnly']
df_selected = data1[selected_columns]
df_selected['Dinner Ratings'].fillna(df_selected['Dinner Ratings'].mean(), inplace=True)
df_selected['Delivery Ratings'].fillna(df_selected['Delivery Ratings'].mean(), inplace=True)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)
user_input = [float(input("Enter Dinner Rating: ")), float(input("Enter Delivery Rating: ")), int(input("Enter isVegOnly (0 or 1): "))]
user_df = pd.DataFrame([user_input], columns=selected_columns)
user_scaled = scaler.transform(user_df)
kmeans = KMeans(n_clusters=5, random_state=0)  # You can choose the number of clusters
kmeans.fit(df_scaled)
user_cluster = kmeans.predict(user_scaled)[0]
data1['Cluster'] = kmeans.predict(df_scaled)
similar_restaurants = data1[data1['Cluster'] == user_cluster]
top_10_restaurants = similar_restaurants[['Name', 'Dinner Ratings', 'Delivery Ratings', 'isVegOnly']].head(10)
print("Top 10 Recommended Restaurants Based on User Input:")
print(top_10_restaurants)


# # Conclusion

# In[65]:


'''In conclusion, we used KMeans clustering to recommend restaurants based on user input for dinner and delivery ratings, along with the preference for vegetarian options. The analysis grouped restaurants into clusters, allowing us to provide the top 10 recommendations tailored to the user's preferences.'''


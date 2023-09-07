#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


with open("/archive/BTC-Daily.csv","r") as f:
    print(f.readlines(5))


# In[6]:


data = pd.read_csv("/archive/BTC-Daily.csv")


# In[7]:


data


# In[5]:


data.describe()


# In[12]:


data_=data.head(26)


# In[7]:


data.tail()


# In[9]:


plt.figure(figsize=(10, 6))
plt.bar(data_['date'], data_['close'], label='Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Bitcoin Daily Closing Price')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# In[10]:


plt.figure(figsize=(10, 6))
plt.plot(data_['date'], data_['high'], label='daily high')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Bitcoin Daily high Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()


# In[17]:


plt.figure(figsize=(10, 6))
plt.plot(data_['date'], data_['low'], label='daily low')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Bitcoin Daily low Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()


# In[11]:


plt.figure(figsize=(10, 6))
plt.bar(data_['date'], data_['Volume BTC'], color='blue', alpha=0.5)
plt.xlabel('Date')
plt.ylabel('Volume BTC')
plt.title('Average Daily Trading Volume of Bitcoin')
plt.xticks(rotation=45)
plt.show()


# In[26]:


plt.figure(figsize=(8, 6))
plt.boxplot(data_['close'])
plt.ylabel('Price (USD)')
plt.title('Box Plot of Bitcoin Closing Prices')
plt.show()


# In[24]:


plt.figure(figsize=(8, 6))
plt.hist(data_['close'], bins=30, color='green', alpha=0.7)
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.title('Histogram of Bitcoin Closing Prices')
plt.show()


# In[21]:


plt.figure(figsize=(8, 6))
plt.scatter(data_['high'], data_['low'], color='orange', alpha=0.7)
plt.xlabel('High Price (USD)')
plt.ylabel('Low Price (USD)')
plt.title('Scatter Plot of High vs. Low Prices')
plt.show()


# In[8]:


data.info()


# In[20]:


sns.pairplot(data)


# In[ ]:





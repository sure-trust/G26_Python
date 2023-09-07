#!/usr/bin/env python
# coding: utf-8

# # Pandas assignment

# # 1st question

# In[1]:


import numpy as np  #1(first question)
import pandas as pd


# In[3]:


random_numbers=np.random.randint(1,100,10)


# In[4]:


random_series=pd.Series(random_numbers)


# In[7]:


random_series.describe()


# # 2nd question

# In[10]:


import pandas as pd


# In[30]:


data={
    'Name':['Deepak.B','Karuna.SH','akshay.CS','Barath.V','Harsha.N'],
    'age':[21, 20, 20, 21, 21],
    'gender': ['Male', 'Female', 'Male', 'Male', 'other'],
    'salary': [60000, 70000, 65000, 55000, 80000]   
}


# In[31]:


df=pd.DataFrame(data)


# In[32]:


print(df)


# In[33]:


print(df.shape)
    


# In[34]:


print(df.index.tolist())


# In[35]:


print(df.columns.tolist())


# # 3rd question

# In[36]:


file_path='/archive/BTC-Daily.csv'


# In[37]:


df = pd.read_csv(file_path)


# In[38]:


print(df.head(10))


# In[39]:


print(df.tail(10))


# In[40]:


random_sample = df.sample(n=10)


# In[41]:


print(random_sample)


# # 4th question

# In[46]:


data={
    'Name':['Deepak.B','Karuna.SH','akshay.CS','Barath.V','Harsha.N'],
    'age':[21, 32, 20, 21, 34],
    'gender': ['Male', 'Female', 'Male', 'Male', 'Female'],
    'salary': [60000, 70000, 65000, 55000, 80000]   
}


# In[47]:


df=pd.DataFrame(data)


# In[48]:


filtered_df = df[(df['age']>30) & (df['gender'] == 'Female')]


# In[49]:


print(filtered_df)


# # 5th question

# In[50]:


grouped_df = df.groupby('gender')['salary'].agg(['mean','median','std'])


# In[51]:


print(grouped_df)


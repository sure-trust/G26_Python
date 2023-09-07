#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.metrics import accuracy_score


# In[35]:


attrdata = pd.read_csv(r"C:\Users\durga\OneDrive\Documents\WA_Fn-UseC_-HR-Employee-Attrition.csv",encoding='iso-8859-1')


# In[36]:


attrdata.head()


# In[37]:


attrdata.drop(0,inplace=True)
attrdata.isnull().sum()


# In[38]:


attrdata.shape


# In[39]:


gender_dict = attrdata["Gender"].value_counts()
gender_dict


# In[40]:


attrdata['Gender'].value_counts().plot(kind='bar',color=['salmon','lightblue'],title="Count of different gender")


# In[41]:


pd.crosstab(attrdata['Gender'],attrdata['YearsAtCompany']).plot(kind="bar",figsize=(10,6))
plt.title("YearsAtCompany vs Gender")
plt.xlabel("YearsAtCompany")
plt.ylabel("No of people who left based on gender")
plt.xticks(rotation=0)


# In[42]:


promoted_dict = attrdata["PerformanceRating"].value_counts()
promoted_dict


# In[43]:


pd.crosstab(attrdata['YearsInCurrentRole'],attrdata['YearsAtCompany']).plot(kind="bar",figsize=(10,6))
plt.title("YearsAtCompany vs YearsInCurrentRole")
plt.xlabel("YearsAtCompany")
plt.ylabel("No of people who left based on gender")
plt.xticks(rotation=0)


# In[44]:


attrdata['PerformanceRating'].value_counts().plot(kind='bar',color=['salmon','lightblue'],title="PerformanceRating")


# In[45]:


func_dict = attrdata["StandardHours"].value_counts()
func_dict


# In[46]:


job_dict = attrdata["YearsInCurrentRole"].value_counts()
job_dict


# In[47]:


attrdata['YearsInCurrentRole'].value_counts().plot(kind='bar',color=['salmon','lightblue'],title="YearsInCurrentRole")


# In[48]:


pd.crosstab(attrdata['YearsInCurrentRole'],attrdata['TotalWorkingYears']).plot(kind="bar",figsize=(10,6))
plt.title("TotalWorkingYears vs YearsInCurrentRole")
plt.xlabel("TotalWorkingYears")
plt.xticks(rotation=0)


# In[52]:


depart_dict = attrdata["Department"].value_counts()
print(depart_dict)


# In[67]:


attrdata['Department'].value_counts().plot(kind='bar',color=['salmon','lightblue'],title="Department")


# In[ ]:





# In[55]:





# In[ ]:





# In[ ]:





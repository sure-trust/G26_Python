#!/usr/bin/env python
# coding: utf-8

# # PROJECT-1

# # Estimating Car Prices

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("C:\\Users\\VutharamalluruBalaji\\Documents\\500rows.csv")
print(data)


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


data.drop("engine_displacement",inplace=True,axis=1)
data.drop("engine_power",inplace=True,axis=1)
data.drop("body_type",inplace=True,axis=1)
data.drop("color_slug",inplace=True,axis=1)
data.drop("stk_year",inplace=True,axis=1)
data.drop("transmission",inplace=True,axis=1)
data.drop("door_count",inplace=True,axis=1)
data.drop("seat_count",inplace=True,axis=1)
data.drop("fuel_type",inplace=True,axis=1)
data.drop("date_created",inplace=True,axis=1)


# In[7]:


data.info()


# In[8]:


data.drop("date_last_seen",inplace=True,axis=1)


# In[9]:


data.info()


# In[10]:


y=data["price_eur"]
y


# In[11]:


x=data.drop("price_eur",axis=1)
x


# In[13]:


from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[14]:


lbenc=LabelEncoder()
for clmn in data.columns:
    if data[clmn].dtype=="object":
        data[clmn]=lbenc.fit_transform(data[clmn])
for clmn in data.columns:
    if data[clmn].dtype=="float64":
        data[clmn]=lbenc.fit_transform(data[clmn])
data.info()
print(data)


# In[15]:


x


# In[16]:


x.info()


# In[17]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25)


# In[18]:


print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)


# In[19]:


le=LabelEncoder()
l1=data["maker"]
l11=le.fit_transform(l1)
l11.reshape(-1,1)
data["maker"]=l11
print(data.info())
print(data)
data.head()


# In[20]:


data.isnull().sum()


# In[21]:


data1=pd.DataFrame(data)
data1.count()


# In[22]:


print("xtrain")
print(xtrain)
print("xtest")
print(xtest)
print("ytrain")
print(ytrain)
print("ytest")
print(ytest)


# In[26]:


xtrain.columns


# In[27]:


lr=LinearRegression()


# In[28]:


lr.fit(xtrain,ytrain)


# In[41]:


y_pred=lr.predict(xtest)


# In[42]:


y_pred


# In[43]:


print("CO-EFFICENTS:",lr.coef_)


# In[44]:


lr.intercept_


# In[45]:


print("Variance score:",lr.score(xtrain,ytrain))


# In[46]:


plt.scatter(lr.predict(xtrain),lr.predict(xtrain)-ytrain,color="blue",s=20,label="Train Data")


# In[50]:


plt.scatter(lr.predict(xtest),lr.predict(xtest)-ytest,color="black",s=20,label="Test Data")


# In[53]:


from sklearn import metrics
print("Mean Absolute Error:",metrics.mean_absolute_error(ytest,y_pred))
print("Mean Squared Error:",metrics.mean_squared_error(ytest,y_pred))
print("Root Mean Squared Error:",np.sqrt(metrics.mean_squared_error(ytest,y_pred)))


# # CONCLUSION

# In[1]:


'''In this project,we have explored the use of machine learning to predict the price of a used car.we used a variety of features,such as the year of manfacture,mileage,model,and condition of the car,to train a machine learning model.The model was able to acheive an accuracy on a test of used cars.'''


# In[ ]:





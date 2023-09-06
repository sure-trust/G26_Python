#!/usr/bin/env python
# coding: utf-8

# # PROJECT-2

# # Drug Activity Prediction

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# In[2]:


data=pd.read_csv("C:\\Users\\VutharamalluruBalaji\\Documents\\Drug prediction.csv")
print(data)


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data.drop("D_4",inplace=True,axis=1)
data.drop("D_40",inplace=True,axis=1)
data.drop("D_42",inplace=True,axis=1)
data.drop("D_46",inplace=True,axis=1)
data.drop("D_47",inplace=True,axis=1)
data.drop("D_48",inplace=True,axis=1)
data.drop("D_58",inplace=True,axis=1)
data.drop("D_59",inplace=True,axis=1)
data.drop("D_63",inplace=True,axis=1)


# In[8]:


data.info()


# In[10]:


col_names=["MOLECULE","Act","D_57","D_60","D_61","D_62","D_64","D_65"]


# In[11]:


data.columns=col_names


# In[12]:


data.head()


# In[13]:


from sklearn.preprocessing import LabelEncoder
lbenc=LabelEncoder()
for clmn in data.columns:
    if data[clmn].dtype=="object":
        data[clmn]=lbenc.fit_transform(data[clmn])
for clmn in data.columns:
    if data[clmn].dtype=="float64":
        data[clmn]=lbenc.fit_transform(data[clmn])
data.info()
print(data)


# In[14]:


x=data.drop("MOLECULE",axis=1)
y=data["MOLECULE"]


# In[15]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[16]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[17]:


rf=RandomForestClassifier()
rf.fit(xtrain,ytrain)


# In[18]:


y_pred=rf.predict(xtest)


# In[19]:


accuracy_score(ytest,y_pred)


# In[20]:


import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[21]:


cnfmat=confusion_matrix(ytest,y_pred)
sns.heatmap(cnfmat,annot=True,fmt=".0f",cmap="plasma")


# In[22]:


plt.xlabel("MOLECULE")
plt.ylabel("Act")
plt.title("histigraph in Drug activity prediction")
plt.hist(data["MOLECULE"],ec='r')


# In[24]:


import seaborn as sns
sns.histplot(data=data,x='Act')


# In[25]:


sns.regplot(data=data,x="MOLECULE",y='Act')


# In[28]:


from sklearn import metrics


# In[29]:


print("Mean Absolute Error:",metrics.mean_absolute_error(ytest,y_pred))
print("Mean Squared Error:",metrics.mean_squared_error(ytest,y_pred))
print("Root Mean Squared Error:",np.sqrt(metrics.mean_squared_error(ytest,y_pred)))


# # CONCLUSION

# In[30]:


'''IN this project,we have explored the use of machine learning to predict the activity of a drug molecule.We used a variety of features,such as the molecular structure the chemicl properties,and the biological targets of the drug molecule,to train a machine learning model.the model was able to acheive an accuracy a test set of drug molecule.''' 


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt


# In[4]:


data_thyroid=pd.read_csv('thyroid data.csv',na_values=["?"])


# In[5]:


data=data_thyroid.copy() #creating the copy of original data


# In[6]:


data


# In[7]:


column_to_remove = 'TBG' #Removing th TBG Because it contains full full null values
data = data.drop(columns=column_to_remove)


# In[8]:


data


# In[9]:


data[data.isnull().any(axis=1)] 


# In[10]:


data.describe()


# # Filling the missing values for numerical datatype

# In[11]:


data.isnull().sum() #Before filling numerical values


# In[12]:


data['age'].mean()


# In[13]:


data['age'].fillna(data['age'].mean(),inplace=True)


# In[14]:


data['TSH'].median()


# In[15]:


data['TSH'].fillna(data['TSH'].median(),inplace=True)


# In[16]:


data['T3'].mean()


# In[17]:


data['T3'].fillna(data['T3'].mean(),inplace=True)


# In[18]:


data.isnull().sum()


# In[19]:


data['TT4'].mean()


# In[20]:


data['TT4'].fillna(data['TT4'].mean(),inplace=True)


# In[21]:


data['T4U'].mean()


# In[22]:


data['T4U'].fillna(data['T4U'].mean(),inplace=True)


# In[23]:


data.isnull().sum()


# In[24]:


data['FTI'].mean()


# In[25]:


data['FTI'].fillna(data['FTI'].mean(),inplace=True)


# In[26]:


data.isnull().sum() #After filling numerical values


# # Filling the missing values for Catagorical datatype

# In[27]:


data.isnull().sum() #Before filling values 


# In[28]:


data['sex'].value_counts().index[0]


# In[29]:


data['sex'].fillna(data['sex'].value_counts().index[0],inplace=True)


# In[30]:


data.isnull().sum() #After filling values


# In[31]:


data.info()


# In[32]:


summary_numerical=data.describe()
print(summary_numerical)


# In[33]:


summary_catagorical=data.describe(include='O')
print(summary_catagorical)


# In[37]:


numerical_columns = data.select_dtypes(include=['float64'])
corr_matrix = numerical_columns.corr()
plt.figure(figsize=(10, 10))
sns.set(font_scale=1.2)
sns.heatmap(corr_matrix, annot=True, cmap='viridis', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[35]:


sns.set(style="whitegrid")
sns.displot(data=data[data['age'].between(1, 100)], x='age', kde=True)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution (1-100 years)')
plt.show()


# In[39]:


cross_table = pd.crosstab(index=data['sex'], columns=data['pregnant'])
plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")
sns.countplot(data=data, x='sex', hue='pregnant')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Count Plot of Sex vs. Pregnant')
plt.legend(title='Pregnant', loc='upper right')
plt.show()


# In[40]:


sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='sex', hue='Class')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Distribution of Class by Sex')
plt.legend(title='Class', loc='upper right')
plt.show()


# In[41]:


plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")
sns.countplot(data=data, x='thyroid_surgery', hue='sex')
plt.xlabel('Thyroid Surgery')
plt.ylabel('Count')
plt.title('Count Plot of Thyroid Surgery and Sex')
plt.legend(title='Sex', loc='upper right')
plt.show()


# In[42]:


filtered_data = data[(data['age'] >= 1) & (data['age'] <= 40)]

plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")

plt.subplot(1, 2, 1)
sns.countplot(data=filtered_data, x='age', hue='sex')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Count Plot of Age vs. Sex (1-40 years)')
plt.legend(title='Sex', loc='upper right')
plt.xticks(rotation=90)

plt.subplot(1, 2, 2)
sns.countplot(data=filtered_data, x='age', hue='thyroid_surgery')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Count Plot of Age vs. Thyroid Surgery (1-40 years)')
plt.legend(title='Thyroid Surgery', loc='upper right')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[55]:


class_mapping = {
    'negative': 0,  # Assign 0 for 'negative'
    'compensated_hypothyroid': 1,  # Assign 1 for 'hypothyroid'
    'primary_hypothyroid': 1,
    'secondary_hypothyroid': 1,
}


# In[57]:


data['Updated_Class']=data['Class'].map(class_mapping)


# In[58]:


data


# In[59]:


new_data=pd.get_dummies(data,drop_first=True)
print(new_data)


# In[60]:


columns_list=list(new_data.columns)
print(columns_list)


# In[61]:


features=list(set(columns_list)-set(['Updated_Class']))
print(features)


# In[62]:


y=new_data['Updated_Class'].values
print(y)


# In[63]:


x=new_data[features].values
print(x)


# In[64]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)


# In[65]:


logistic=LogisticRegression()


# In[66]:


logistic = LogisticRegression(max_iter=1000)
logistic.fit(train_x, train_y)


# In[67]:


logistic.coef_


# In[68]:


logistic.intercept_


# In[69]:


prediction=logistic.predict(test_x)
print(prediction)


# In[70]:


from sklearn.metrics import confusion_matrix
confusion_matrix_result = confusion_matrix(test_y, prediction)
print(confusion_matrix_result)


# In[71]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y, prediction)
print(accuracy)


# In[72]:


print('Misclassified samples:%d'%(test_y!=prediction).sum())


# In[79]:


feature_names = features  
coefficients = logistic.coef_[0]
coefficients_percent = (coefficients / coefficients.sum()) * 100
feature_coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient (%)': coefficients_percent})
feature_coefficients_df = feature_coefficients_df.sort_values(by='Coefficient (%)', ascending=False)
plt.figure(figsize=(12, 6))
plt.bar(feature_coefficients_df['Feature'], feature_coefficients_df['Coefficient (%)'])
plt.xlabel('Feature')
plt.ylabel('Coefficient (%)')
plt.title('Feature Coefficients for Classification (as Percentages)')
plt.xticks(rotation=90)
plt.show()


# # Conclusion

# In[3]:


''' In Conclusion we can say that more people thyroid result is negative less than 20% people tested positve for the thyroid desiese ans even less people got thyroid surgery for the thyroid disease,logistic regression is the suitable for the classification for the thyroid disease '''


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[99]:


df = pd.read_csv(r"C:\Users\durga\OneDrive\Desktop\New folder\1651277648862_healthinsurance.csv",encoding='iso-8859-1')
df.head()


# In[100]:


df.info()


# In[101]:


df.shape


# In[102]:


df.describe()


# In[103]:


df.isna().sum()


# In[104]:


orig_df=df.copy()


# In[105]:


from sklearn import preprocessing
columns = ['hereditary_diseases','job_title','city','sex']  # columns names where transform is required
for X in columns:
  exec(f'le_{X} = preprocessing.LabelEncoder()')  #create label encoder with name "le_X", where X is column name
  exec(f'df.{X} = le_{X}.fit_transform(df.{X})')  #execute fit transform for column X with respective lable encoder "le_X", where X is column name
df.head() 


# In[106]:


df.skew()


# In[107]:


updated_df = df
updated_df['age']=updated_df['age'].fillna(updated_df['age'].mean())

updated_df.info()


# In[108]:


updated_df = df
updated_df['bmi']=updated_df['bmi'].fillna(updated_df['bmi'].mean())

updated_df.info()


# In[109]:


df.duplicated().sum()


# In[110]:


df[df.duplicated()]


# In[111]:


updated_df = updated_df.drop_duplicates()
updated_df.duplicated().sum()


# In[112]:


plt.figure(figsize=(10, 6))
sns.histplot(x='claim', data=updated_df, kde=True)
plt.axvline(updated_df.claim.mean(), color='r', linestyle='--', label='Mean')
plt.axvline(updated_df.claim.median(), color='g', linestyle='--', label='Median')
plt.title('Claim Distribution')
plt.xlabel('Claim')
plt.ylabel('Frequency')
plt.show()
print(f'Skewness: {updated_df["claim"].skew()}')
print(f'Mean: {(updated_df["claim"].mean())}')
print(f'Median: {updated_df["claim"].median()}')


# In[113]:


plt.figure(figsize=(20, 6))
sns.histplot(x='age', data=updated_df, kde=True)
plt.axvline(updated_df.age.mean(), color='r', linestyle='--', label='Mean')
plt.axvline(updated_df.age.median(), color='g', linestyle='--', label='Median')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')


# In[114]:


plt.figure(figsize=(20, 6))
sns.scatterplot(x='age', y='claim', data=updated_df, hue='smoker')
plt.title('Age vs Claim')
plt.xlabel('Age')
plt.ylabel('Claim')
plt.show()
print(f'Skewness: {updated_df["age"].skew()}')
print(f'Mean: {(updated_df["age"].mean()):.0f}')
print(f'Median: {updated_df["age"].median()}')
print(f'Minimum Age: {updated_df["age"].min()}')
print(f'Maximum Age: {updated_df["age"].max()}')


# In[115]:


plt.figure(figsize=(15, 6))
fig = sns.countplot(x='sex', data=updated_df)
for p in fig.patches:
    height = p.get_height()
    fig.text(p.get_x() + p.get_width()/2.,height + 3, '{:1.0f}'.format(height), ha="center")
plt.title('Gender Count')
plt.xlabel('Gender')
plt.ylabel('Count')


# In[116]:


sns.boxplot(x='sex', y='claim', data=updated_df)
plt.title('Claim Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Claim')
plt.show()


# In[117]:


fig = sns.countplot(x='sex', data=updated_df, hue='smoker')
for p in fig.patches:
    height = p.get_height()
    fig.text(p.get_x() + p.get_width()/2.,height + 3, '{:1.0f}'.format(height), ha="center")
plt.title('Gender Count by Smoker')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[118]:


round(updated_df.groupby('sex').claim.mean(), 0).reset_index()


# In[119]:


sns.boxplot(x='no_of_dependents', y='claim', data=updated_df)
plt.title('Claim Distribution by no. of dependents')
plt.xlabel('No. Of Dependents')
plt.ylabel('Claim')
plt.show()
print(f'Mean: {(updated_df["no_of_dependents"].mean()):.0f}')


# In[120]:


corr_num = updated_df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_num, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[121]:


features = list(updated_df.columns[:12])
print("features:", features, sep="\n")


# In[122]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

y = updated_df["claim"]
X = updated_df[features]

selector = SelectKBest(f_regression, k=8)
selector.fit(X, y)

X_new = selector.transform(X)
print(updated_df.columns[selector.get_support(indices=True)].tolist())
final_features = updated_df.columns[selector.get_support(indices=True)].tolist()


# In[123]:


df_train_val, df_test = train_test_split(updated_df, test_size=0.04, random_state=23)
df_train, df_val = train_test_split(df_train_val, test_size=0.1, random_state=23)
print(f'df_train proportion: {len(df_train) / len(updated_df):.2f}')
print(f'df_val proportion: {len(df_val) / len(updated_df):.2f}')
print(f'df_test proportion: {len(df_test) / len(updated_df):.2f}')


# In[124]:


plt.figure(figsize=(25, 6))
plt.subplot(1, 3, 1)
sns.boxplot(x='age', data=df_train)
plt.title('Age Distribution')
plt.xlabel('Frequency')
plt.subplot(1, 3, 2)
sns.boxplot(x='bmi', data=df_train)
plt.title('BMI Distribution')
plt.xlabel('Frequency')
plt.subplot(1, 3, 3)
sns.boxplot(x='claim', data=df_train)
plt.title('Claim Distribution')
plt.xlabel('Frequency')
plt.show()


# In[125]:


def skewed_dis(df, col):
    IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
    lower_bound = df[col].quantile(0.25) - (IQR * 1.5)
    upper_bound = df[col].quantile(0.75) + (IQR * 1.5)
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)] 
outliers = skewed_dis(df_train, 'claim')
print(f"Outlier percentage: {(len(outliers) / len(df_train) * 100):.1f}%")
outliers


# In[126]:


# Training Set
X_train = df_train[final_features]
y_train = df_train.claim

# Validation Set
X_val = df_val[final_features]
y_val = df_val.claim


# In[127]:


featuresx = final_features
targetx = 'claim'

# Creating a scatter matrix
pd.plotting.scatter_matrix(updated_df[featuresx + [targetx]], figsize=(10, 10))
plt.show()


# # Linear  Regression

# In[128]:


model_1 = LinearRegression()


# In[129]:


model_1.fit(X_train, y_train)


# In[130]:


training_score_1 = model_1.score(X_train, y_train)


# In[131]:


y_train_pred_1 = model_1.predict(X_train)
training_mae_1 = mean_absolute_error(y_train, y_train_pred_1)
print("Training Score: ", training_score_1)
print("Training MAE: ", training_mae_1)


# In[132]:


model_1_v = LinearRegression()
model_1_v.fit(X_val, y_val)


# In[133]:


training_score_1_v = model_1_v.score(X_val, y_val)
y_train_pred_1_v = model_1_v.predict(X_val)
training_mae_1_v = mean_absolute_error(y_val, y_train_pred_1_v)
print("Training Score: ", training_score_1_v)
print("Training MAE: ", training_mae_1_v)


# In[134]:


df_test


# In[135]:


# Split Features and Target
X_test = df_test[final_features]
y_test = df_test.claim

predictions = df_test.copy()
predictions.head(10)


# # Logistic regression

# In[136]:


import random
import numpy as np

outcomes = ["Yes", "No"]
random.seed(30)
claim = np.array([random.randint(30,100) for _ in range(10)]).reshape(-1,1)
out_comes = [random.choice(outcomes) for _ in range(10)]


# In[137]:


from sklearn.linear_model import LogisticRegression


# In[138]:


lr = LogisticRegression()
lr.fit(claim, out_comes)


# In[139]:


from sklearn.model_selection import train_test_split as tts


# In[140]:


xtrain, xtest, ytrain, ytest = tts(df.drop("claim",axis= 1), df["claim"], train_size = 0.75)


# In[141]:


xtrain.shape, ytest.shape


# In[142]:


lr.coef_


# In[143]:


lr.intercept_


# In[144]:


lr.predict_proba([[40]])


# In[145]:


lr.predict([[40]])


# In[146]:


xtest


# # Polynomial Regression

# In[152]:


from sklearn.preprocessing import PolynomialFeatures


# In[154]:


p = PolynomialFeatures(degree=2)
x_copy = p.fit_transform(claim)


# In[160]:


lr_polynomial = LinearRegression()
lr_polynomial.fit(x_copy, claim)


# In[164]:


plt.plot(x_copy[:,1], lr_polynomial.predict(x_copy))
plt.show()


# In[166]:


lr_polynomial.score(x_copy,claim)


# In[167]:


lr_polynomial.intercept_


# In[168]:


lr_polynomial.coef_


# In[ ]:





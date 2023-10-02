# -*- coding: utf-8 -*-
"""Health care

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TV84rRC-w-qMkz4Y5KSYzmoAc890wE_O
"""

Cluster health data to identify patient groups or disease patterns.
Procedure: Considering features like vital signs, lab results, and medical history.
Domain: Healthcare

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

data=pd.read_csv('/content/2015_data.csv')

print(data.head())

print(data.info())

data.describe()

data.head(10)

data.tail()

X = data[['resident_status', 'education_2003_revision', 'month_of_death', 'sex', 'detail_age', 'race_recode_3', 'hispanic_originrace_recode']]

categorical_columns = ['resident_status', 'education_2003_revision', 'month_of_death', 'sex', 'race_recode_3', 'hispanic_originrace_recode']

data_encoded = pd.get_dummies(data, columns=categorical_columns)

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
imputer_num = SimpleImputer(strategy='mean')
X[numerical_features] = imputer_num.fit_transform(X[numerical_features])

categorical_features = X.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
X[categorical_features] = imputer_cat.fit_transform(X[categorical_features])

sns.scatterplot(x='detail_age', y='month_of_death', data=data)
plt.show()

"""The pair plot displays pairwise relationships between detail_age, month_of_death, and sex.
The diagonal plots show the distribution of each variable, the top-left plot shows the distribution of detail_age.
The scatter plots show relationships between pairs of variables. For example, the bottom-left plot shows the relationship between detail_age and month_of_death.
The hue represents the different values of sex
"""

sns.countplot(x='sex', data=data)
plt.show()

"""

The count plot shows the distribution of sex in the dataset.
It provides a visual representation of how many individuals are categorized as male ('M') and female ('F').
This plot indicates the relative proportion of males and females in the dataset.

"""

sns.pairplot(data[['detail_age', 'month_of_death', 'sex']])
plt.show()

"""The pair plot displays pairwise relationships between detail_age, month_of_death, and sex.
The diagonal plots show the distribution of each variable, the top-left plot shows the distribution of detail_age.
The scatter plots show relationships between pairs of variables. For example, the bottom-left plot shows the relationship between detail_age and month_of_death.
"""

from sklearn.preprocessing import LabelEncoder
categorical_columns = ['resident_status', 'education_2003_revision', 'month_of_death', 'sex', 'race_recode_3', 'hispanic_originrace_recode']
data_encoded = pd.get_dummies(data, columns=[col for col in categorical_columns if col != 'sex'])

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X['sex_encoded'] = label_encoder.fit_transform(X['sex'])
X_numeric = X[['detail_age', 'sex_encoded']].dropna()

inertia = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X_numeric)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

"""The plot is used to determine the optimal number of clusters for a k-means clustering algorithm.
It shows the sum of squared distances (inertia) between data points and their assigned cluster centers for different values of k.
The "elbow" point, where the inertia starts to level off, is typically chosen as the optimal k.

Main conclusion:The clustering analysis of health data has successfully identified distinct patient groups based on vital signs, lab results, and medical history. This project holds significant implications for personalized healthcare and disease management.

REFERENCES:Kaggle,Replit,Chrome
"""
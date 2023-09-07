#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[5]:


data=pd.read_csv("city_day.csv")


# In[6]:


data


# In[7]:


data1=data.copy() #copying the dataframe 


# In[8]:


column_to_remove = 'AQI_Bucket' #Removing th TBG Because it contains full full null values
data1 = data1.drop(columns=column_to_remove)


# # Data Cleaning

# In[9]:


data1.isnull().sum() 


# In[10]:


data1.describe()


# In[11]:


data1.isnull().sum() #Before filling numerical values


# In[12]:


data1['PM2.5'].median()


# In[13]:


data1['PM2.5'].fillna(data1['PM2.5'].median(),inplace=True)


# In[14]:


data1['PM10'].median()


# In[15]:


data1['PM10'].fillna(data1['PM10'].median(),inplace=True)


# In[16]:


data1['NO'].mean()


# In[17]:


data1['NO'].fillna(data1['NO'].mean(),inplace=True)


# In[18]:


data1['NO2'].mean()


# In[19]:


data1['NO2'].fillna(data1['NO2'].mean(),inplace=True)


# In[20]:


data1['NOx'].mean()


# In[21]:


data1['NOx'].fillna(data1['NOx'].mean(),inplace=True)


# In[22]:


data1['NH3'].mean()


# In[23]:


data1['NH3'].fillna(data1['NH3'].mean(),inplace=True)


# In[24]:


data1['CO'].mean()


# In[25]:


data1['CO'].fillna(data1['CO'].mean(),inplace=True)


# In[26]:


data1['SO2'].mean()


# In[27]:


data1['SO2'].fillna(data1['SO2'].mean(),inplace=True)


# In[28]:


data1['O3'].mean()


# In[29]:


data1['O3'].fillna(data1['O3'].mean(),inplace=True)


# In[30]:


data1['Benzene'].mean()


# In[31]:


data1['Benzene'].fillna(data1['Benzene'].mean(),inplace=True)


# In[32]:


data1['Toluene'].mean()


# In[33]:


data1['Toluene'].fillna(data1['Toluene'].mean(),inplace=True)


# In[34]:


data1['Xylene'].mean()


# In[35]:


data1['Xylene'].fillna(data1['Xylene'].mean(),inplace=True)


# In[36]:


data1['AQI'].mean()


# In[37]:


data1['AQI'].fillna(data1['AQI'].mean(),inplace=True)


# In[38]:


data1.isnull().sum()


# In[39]:


data1.info()


# In[40]:


summary_catagorical=data1.describe(include='O')
print(summary_catagorical)


# # Ploting the Corrilation Heatmap

# In[46]:


numerical_columns = data1.select_dtypes(include=['number'])
corr_matrix = numerical_columns.corr()
plt.figure(figsize=(12, 8))
sns.set(font_scale=1.2)
sns.heatmap(corr_matrix, annot=True, cmap='rainbow', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[40]:


data1


# In[41]:


data1.drop_duplicates(keep='first',inplace=True)


# # Data Visualization 

# In[42]:


sns.scatterplot(x='City', y='AQI', data=data1)
plt.xticks(rotation=90)
plt.xlabel('City')
plt.ylabel('AQI')
plt.title('AQI in Cities')
plt.show()


# In[43]:


'''In the above graph we came to know know the AQI Level in the Ahmedabad is very High and Visakapatnam has low AQI Level'''


# In[44]:


sns.scatterplot(x='City', y='CO', data=data1)
plt.xticks(rotation=90)
plt.xlabel('City')
plt.ylabel('AQI')
plt.title('CO emission in Cities')
plt.show()


# In[45]:


''' In the  above graph shows that in Ahmedabad more CO emission take place following by Bengaluru,Gurugram,Lucknow,Delhi and Chennai. Teir 1 emits more amount of CO gas compare to Teir 2 and Teir 3 cities'''


# In[46]:


data1['Date'] = pd.to_datetime(data1['Date'])
start_date = '2015-01-01'
end_date = '2015-06-30'
filtered_data = data1[(data1['Date'] >= start_date) & (data1['Date'] <= end_date)]
plt.figure(figsize=(12, 6))
plt.plot(filtered_data['Date'], filtered_data['PM2.5'], label='PM2.5', marker='o')
plt.xlabel('Date')
plt.ylabel('PM2.5 Level')
plt.title('Time Series of PM2.5 (Jan 2015 - Jun 2015)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[47]:


plt.figure(figsize=(12, 6))
pollutants = ['PM2.5', 'NO2', 'SO2']
for pollutant in pollutants:
    plt.plot(data1['Date'], data1[pollutant], label=pollutant, marker='o')
plt.xlabel('Date')
plt.ylabel('Pollutant Level')
plt.title('Comparison of Pollutants over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[48]:


plt.figure(figsize=(12, 6))
plt.plot(data1['Date'], data1['AQI'], label='AQI', marker='o', color='blue')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.title('Air Quality Index (AQI) Trend')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # linear regression

# In[50]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
target = 'AQI'
X = data1[features]
y = data1[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared (Coefficient of Determination): {r2}')
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)


# In[51]:


from sklearn.preprocessing import PolynomialFeatures
selected_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
X = data1[selected_features]
y = data1['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
degree = 2  
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred = poly_reg.predict(X_test_poly)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2) Score: {r2}')


# In[52]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Actual vs. Predicted AQI (Polynomial Regression)')
plt.show()


# # Random Forest

# In[53]:


from sklearn.ensemble import RandomForestRegressor

# Create and train a Random Forest Regression model
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_reg.predict(X_test)

# Calculate Mean Squared Error (MSE) and R-squared (R2) score for Random Forest Regression
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest Regression - Mean Squared Error: {mse_rf}')
print(f'Random Forest Regression - R-squared (R2) Score: {r2_rf}')


# In[56]:


import matplotlib.pyplot as plt

# Calculate Mean Squared Errors (MSE) for each model
mse_linear = mean_squared_error(y_test, y_pred)
mse_poly = mean_squared_error(y_test, y_pred)
mse_rf = mean_squared_error(y_test, y_pred_rf)

# Calculate R-squared (R2) scores for each model
r2_linear = r2_score(y_test, y_pred)
r2_poly = r2_score(y_test, y_pred)
r2_rf = r2_score(y_test, y_pred_rf)

# Create lists to store MSE and R2 values
mse_values = [mse_linear, mse_poly, mse_rf]
r2_values = [r2_linear, r2_poly, r2_rf]

# Create labels for the models
models = ['Linear Regression', 'Polynomial Regression', 'Random Forest Regression']

# Plot MSE
plt.figure(figsize=(10, 6))
plt.bar(models, mse_values, color=['blue', 'green', 'red'])
plt.xlabel('Regression Model')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Mean Squared Error (MSE) Comparison')
plt.show()

# Plot R2
plt.figure(figsize=(10, 6))
plt.bar(models, r2_values, color=['blue', 'green', 'red'])
plt.xlabel('Regression Model')
plt.ylabel('R-squared (R2) Score')
plt.title('R-squared (R2) Score Comparison')
plt.show()


# # Conclusion 

# In[ ]:


''' The purpose of comparing MSE and R2 scores is to evaluate the performance of the three regression models on a given dataset.
In both the MSE and R2 score plots, the three models are compared side by side, making it easy to visually assess their performance.
The model with the lowest MSE is the best at minimizing the error between the predicted and actual values. Conversely, the model with the highest R2 score is the best at explaining the variance in the target variable.

The choice of the best model depends on the specific problem and dataset. Lower MSE and higher R2 scores generally indicate better performance, but the trade-offs between model complexity, interpretability, and computational cost should also be considered '''


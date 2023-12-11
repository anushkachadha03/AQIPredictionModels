#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the dataset
data = pd.read_csv('***')

# Extract features (X) and target variable (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, 7].values
y = y.reshape(-1, 1)

# Handling missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
y = imputer.fit_transform(y)

imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X[:, 1:8] = imputer2.fit_transform(X[:, 1:8])

# Split the dataset into training and test sets
xt = X[1:394, 0:8]
yt = y[1:394, 0]
xtest = X[394:400, 0:8]
ytest = y[394:400, 0]

# Train Random Forest Regression model
rf_regressor = RandomForestRegressor(random_state=0)
rf_regressor.fit(xt[:, 1:8], yt)

# Predict using the trained model
yp = rf_regressor.predict(xtest[:, 1:8])

# Visualize the results
plt.scatter(xtest[:, 0], ytest, color='red', label='Actual Values')
plt.scatter(xtest[:, 0], yp, color='blue', label='Predicted Values')
plt.legend()
plt.title('Random Forest Regression - Actual vs Predicted')
plt.show()

# Calculate RMSE
mse = mean_squared_error(ytest, yp)
rmse = sqrt(mse)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")


# In[ ]:





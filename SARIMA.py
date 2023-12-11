#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pmdarima import auto_arima
import numpy as np


# In[3]:


# Load dataset
df = pd.read_csv('***', parse_dates=['DATE'], dayfirst=True)
df.set_index('DATE', inplace=True)


# In[23]:


# Select specific pollutant column
pollutant_column = 'SO2'
y = df[pollutant_column]

# Apply log filter to eliminate irregular trend and multiplicative seasonality pattern
df_log = np.log1p(y)

# Try different values for m
for m in range(1, 13):
    auto_model = auto_arima(df_log, seasonal=True, m=m, suppress_warnings=True, stepwise=True)
    print(f"m={m}: AIC={auto_model.aic()}")
    


# In[25]:


# Perform a grid search for SARIMA parameters
auto_model = auto_arima(df_log, seasonal=True, m=8, suppress_warnings=True, stepwise=True, trace=True)
print(auto_model.summary())


# In[30]:


from matplotlib import pyplot as plt
import statsmodels.api as sm

#lags=30, for daily data with monthly seasonality
# ACF plot 
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# ACF plot
lags_acf = 40  # Adjust as needed
sm.graphics.tsa.plot_acf(df_log, lags=lags_acf, ax=ax[0])
ax[0].set_title('Autocorrelation Function (ACF)')

# PACF plot
lags_pacf = 40  # Adjust as needed
sm.graphics.tsa.plot_pacf(df_log, lags=lags_pacf, ax=ax[1])
ax[1].set_title('Partial Autocorrelation Function (PACF)')

plt.show()


# In[7]:


from matplotlib import pyplot as plt
import statsmodels.api as sm
# Define different sets of SARIMA parameters for each column
sarima_params = {
    'PM2.5': {'order': (1, 1, 1), 'seasonal_order': (0, 0, 1, 11)},
    'PM10': {'order': (1, 1, 1), 'seasonal_order': (1, 0, 1, 8)},
    'NO2': {'order': (2, 1, 2), 'seasonal_order': (1, 0, 0, 3)},
    'NH3': {'order': (2, 1, 1), 'seasonal_order': (2, 0, 0, 10)},
    'SO2': {'order': (1, 1, 1), 'seasonal_order': (1, 0, 1, 8)},
    'CO': {'order': (2, 1, 5), 'seasonal_order': (1, 0, 0, 10)},
    'OZONE': {'order': (0, 1, 2), 'seasonal_order': (1, 0, 0, 10)}
}

# Initialize a dictionary to store forecast results for each column
forecast_results = {}

# Iterate over each column and its specified SARIMA parameters
for column, params in sarima_params.items():
    # Select specific pollutant column
    y = df[column]

    # Apply log filter to eliminate irregular trend and multiplicative seasonality pattern
    df_log = np.log1p(y)

    # Fit the SARIMA model with specified parameters
    sarima_model = sm.tsa.SARIMAX(df_log, order=params['order'], seasonal_order=params['seasonal_order'])
    sarima_result = sarima_model.fit()

    # Make predictions for the last 5 rows
    forecast_steps = 5  # Number of steps to forecast
    forecast = sarima_result.get_forecast(steps=forecast_steps)

    # Get confidence intervals for the forecast
    confidence_intervals = forecast.conf_int()

    # Convert the predictions and confidence intervals back to the original scale
    forecast_values = np.expm1(forecast.predicted_mean)
    lower_confidence = np.expm1(confidence_intervals.iloc[:, 0])
    upper_confidence = np.expm1(confidence_intervals.iloc[:, 1])

    # Store the results in the dictionary
    forecast_results[column] = {
        'forecast_values': forecast_values,
        'lower_confidence': lower_confidence,
        'upper_confidence': upper_confidence
    }

# Plot the original data and SARIMA forecasts with confidence intervals
plt.figure(figsize=(12, 8))

for column in forecast_results:
    original_column = column.replace('_log', '')
    plt.plot(df[original_column].index, df[original_column], label=f'Original {original_column} Data')

    # Plot forecast values
    plt.plot(df[original_column].index[-forecast_steps:], forecast_results[column]['forecast_values'],
             label=f'SARIMA Forecast for {original_column}')

    # Plot confidence intervals
    plt.fill_between(df[original_column].index[-forecast_steps:],
                     forecast_results[column]['lower_confidence'],
                     forecast_results[column]['upper_confidence'],
                     alpha=0.3, label=f'95% CI for {original_column}')

plt.title('SARIMA Forecast for Pollutant Concentrations')
plt.xlabel('Date')
plt.ylabel('Original Values')
plt.legend()
plt.show()


# In[12]:


# Print the original and predicted values for the last 5 rows for each column
for column in forecast_results:
    original_column = column.replace('_log', '')
    
    # Extract the original values for the last 5 rows
    original_values_last_5 = df[original_column].tail(5).values
    
    # Extract the predicted values for the last 5 rows
    predicted_values_last_5 = forecast_results[column]['forecast_values']
    
    print(f'Original values for {original_column} (last 5 rows):')
    print(original_values_last_5)
    
    print(f'Predicted values for {column} (last 5 rows):')
    print(predicted_values_last_5)
    
    print('\n')


# In[13]:


from sklearn.metrics import mean_squared_error
from math import sqrt

meanSquaredError=mean_squared_error(original_values_last_5, predicted_values_last_5)
print("MSE:", meanSquaredError)
rmse_dt = sqrt(meanSquaredError)
print("RMSE:", rmse_dt)

plt.title('RMSE: %0.2f'% rmse_dt)
plt.show()


# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 17:10:18 2025

@author: Aman Prakash Shah
"""

#%%
# Import necessary libraries for data analysis, visualization, and time series modeling
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

#%%
# Load the dataset containing exchange rate and stock price information
df = pd.read_csv('Book1.csv')

# Convert the 'Date' column to datetime format for proper time series handling
# Note: There appears to be a typo in the format string - should likely be '%d-%m-%Y'
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Set the 'Date' column as the index for time series operations
df.set_index('Date', inplace=True)

# Display the first few rows of the dataset to verify loading and preprocessing
print(df.head())

# Plot the exchange rate and stock prices over time to visualize trends and relationships
plt.figure(figsize=(14, 8))
plt.plot(df['Close USD to INR'], label='USD to INR Exchange Rate')
plt.plot(df['Close APPL (USD)'], label='Apple Stock Price (USD)')
plt.plot(df['Close TCS (USD)'], label='TCS Stock Price (USD)')
plt.title('Exchange Rate and Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

#%%
# Calculate correlations between exchange rates and stock prices to quantify relationships
correlation_matrix = df[['Close USD to INR', 'Close APPL (USD)', 'Close TCS (USD)']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap for easier interpretation
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix: Exchange Rate and Stock Prices')
plt.show()

#%%
# Prepare data for Vector Autoregression (VAR) model
# VAR models capture linear interdependencies among multiple time series
var_data = df[['Close USD to INR', 'Close APPL (USD)', 'Close TCS (USD)']].dropna()

# Check for stationarity using Augmented Dickey-Fuller test
# Stationarity is required for VAR modeling
for column in var_data.columns:
    result = adfuller(var_data[column])
    print(f'ADF Statistic for {column}: {result[0]}')
    print(f'p-value for {column}: {result[1]}')

# Since data is not stationary (p-values > 0.05), difference the data
var_data_diff = var_data.diff().dropna()

# Fit the VAR model with optimal lag order selected by AIC criterion
model_var = VAR(var_data_diff)
results_var = model_var.fit(maxlags=5, ic='aic')
print(results_var.summary())

# Forecast the next 30 days using the fitted VAR model
lag_order = results_var.k_ar
forecast_var = results_var.forecast(var_data_diff.values[-lag_order:], steps=30)
print("VAR Forecast:")
print(forecast_var)

# Plot the forecast results to visualize future predictions
forecast_index = pd.date_range(start=var_data_diff.index[-1], periods=31, freq="D")[1:]
forecast_df = pd.DataFrame(forecast_var, index=forecast_index, columns=var_data_diff.columns)

plt.figure(figsize=(14, 8))
plt.plot(var_data_diff['Close USD to INR'], label='Historical Exchange Rate (Differenced)')
plt.plot(forecast_df['Close USD to INR'], label='Forecast Exchange Rate (Differenced)')
plt.title('USD to INR Exchange Rate Forecast (VAR Model)')
plt.xlabel('Date')
plt.ylabel('Differenced Exchange Rate')
plt.legend()
plt.show()

#%% 
# Split data into train and test (e.g., 80-20 split)
train_size = int(len(var_data_diff) * 0.8)
train, test = var_data_diff.iloc[:train_size], var_data_diff.iloc[train_size:]

# Fit VAR model on training data
model_var = VAR(train)
results_var = model_var.fit(maxlags=5, ic='aic')

# Forecast on test data
lag_order = results_var.k_ar
forecast_var = results_var.forecast(train.values[-lag_order:], steps=len(test))

# Convert forecasts to DataFrame
forecast_df = pd.DataFrame(forecast_var, index=test.index, columns=test.columns)

# Calculate MSE and MAE for each variable (e.g., 'Close USD to INR')
mse_var = mean_squared_error(test['Close USD to INR'], forecast_df['Close USD to INR'])
mae_var = mean_absolute_error(test['Close USD to INR'], forecast_df['Close USD to INR'])

print(f"VAR Model - MSE: {mse_var:.4f}, MAE: {mae_var:.4f}")

#%%
# Perform Granger Causality tests to examine predictive relationships between variables
# Tests whether one time series can predict another
granger_test = grangercausalitytests(var_data_diff[['Close USD to INR', 'Close APPL (USD)']], maxlag=5)

# Calculate rolling correlation to examine how relationships change over time
rolling_corr = df['Close USD to INR'].rolling(window=30).corr(df['Close APPL (USD)'])


#%%
# Plot rolling correlation to visualize changing relationships
plt.figure(figsize=(12, 6))
plt.plot(rolling_corr, label='Rolling Correlation (USD/INR vs Apple Stock Price)')
plt.title('Rolling Correlation: Exchange Rate vs Apple Stock Price')
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.legend()
plt.show()

#%%
# Calculate rolling volatility to measure risk over time
df['Exchange Rate Volatility'] = df['Close USD to INR'].rolling(window=30).std()
df['Apple Stock Volatility'] = df['Close APPL (USD)'].rolling(window=30).std()

# Plot volatility to compare risk profiles
plt.figure(figsize=(12, 6))
plt.plot(df['Exchange Rate Volatility'], label='Exchange Rate Volatility (USD/INR)')
plt.plot(df['Apple Stock Volatility'], label='Apple Stock Volatility')
plt.title('Volatility of Exchange Rate and Apple Stock Price')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()

#%%
# Calculate daily returns for risk-return analysis
df['Exchange Rate Returns'] = df['Close USD to INR'].pct_change()
df['Apple Stock Returns'] = df['Close APPL (USD)'].pct_change()
df['TCS Stock Returns'] = df['Close TCS (USD)'].pct_change()

# Plot correlation heatmap of returns to understand co-movements
returns_df = df[['Exchange Rate Returns', 'Apple Stock Returns', 'TCS Stock Returns']].dropna()
sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap: Returns of Exchange Rate and Stock Prices')
plt.show()

#%%
# Conduct event study to analyze impact of specific market events
event_date = pd.to_datetime('2015-08-24')  # Example: Global market crash
window = 30  # Days before and after the event

# Plot exchange rate and stock prices around the event date
plt.figure(figsize=(12, 6))
plt.plot(df.loc[event_date - pd.Timedelta(days=window):event_date + pd.Timedelta(days=window), 'Close USD to INR'], 
         label='USD to INR Exchange Rate')
plt.plot(df.loc[event_date - pd.Timedelta(days=window):event_date + pd.Timedelta(days=window), 'Close APPL (USD)'],
         label='Apple Stock Price')
plt.axvline(event_date, color='red', linestyle='--', label='Event Date')
plt.title('Event Study: Impact on Exchange Rate and Stock Prices')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

#%%
# Train Random Forest model for exchange rate prediction using machine learning
X = df[['Close APPL (USD)', 'Close TCS (USD)']].shift(1).dropna()  # Lagged features
y = df['Close USD to INR'].iloc[1:]  # Target variable

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X, y)

# Plot feature importance to understand predictive power of each variable
feature_importance = model_rf.feature_importances_
plt.figure(figsize=(8, 4))
plt.bar(X.columns, feature_importance)
plt.title('Feature Importance for Exchange Rate Prediction')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Perform scenario analysis: What if Apple stock price drops by 10%?
scenario_data = X.copy()
scenario_data['Close APPL (USD)'] = scenario_data['Close APPL (USD)'] * 0.9

# Predict exchange rates under the scenario
scenario_predictions = model_rf.predict(scenario_data)

# Plot scenario predictions versus actual values
plt.figure(figsize=(12, 6))
plt.plot(y.index, y, label='Actual Exchange Rate')
plt.plot(y.index, scenario_predictions, label='Scenario: Apple Stock Price Drops 10%')
plt.title('Scenario Analysis: Impact of Apple Stock Price Drop on Exchange Rate')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend()
plt.show()

#%%


# Prepare features (X) and target (y)
X = df[['Close APPL (USD)', 'Close TCS (USD)']].shift(1).dropna()
y = df['Close USD to INR'].iloc[1:]

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Predict on test set
y_pred = model_rf.predict(X_test)

# Calculate MSE and MAE
mse_rf = mean_squared_error(y_test, y_pred)
mae_rf = mean_absolute_error(y_test, y_pred)

print(f"Random Forest - MSE: {mse_rf:.4f}, MAE: {mae_rf:.4f}")

#%%

# Analyze portfolio performance combining Apple stock and USD/INR exchange rate
portfolio_returns = df['Apple Stock Returns'] + df['Exchange Rate Returns']

# Plot portfolio returns to assess risk-return profile
plt.figure(figsize=(12, 6))
plt.plot(portfolio_returns, label='Portfolio Returns (Apple + USD/INR)')
plt.title('Portfolio Risk-Return Profile')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.show()
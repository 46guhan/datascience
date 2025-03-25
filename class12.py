import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Generate sample time series data
np.random.seed(42)
time_index = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
values = np.random.randn(len(time_index))
ts_data = pd.Series(values, index=time_index)
print(ts_data)
# Plot the time series data
plt.figure(figsize=(10, 6))
plt.plot(ts_data)
plt.title('Sample Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Split the data into training and testing sets
train_size = int(len(ts_data) * 0.8)
train_data, test_data = ts_data[:train_size], ts_data[train_size:]

# Fit ARIMA model
order = (2, 1, 1)  # ARIMA order (p, d, q)
model = ARIMA(train_data, order=order)
model_fit = model.fit()

# Forecast future values
forecast_steps = len(test_data)
forecast = model_fit.forecast(steps=forecast_steps)

# Plot actual vs. forecasted values
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data, label='Actual')
plt.plot(test_data.index, forecast, color='red', label='Forecast')
plt.title('ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate model performance
mse = mean_squared_error(test_data, forecast)
print('Mean Squared Error (MSE):', mse)
 
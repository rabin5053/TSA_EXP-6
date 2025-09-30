Ex.No: 6 HOLT WINTERS METHOD

Date:30-09-2025

AIM:

To implement Holt-Winters model on National stocks exchange Data Set and make future predictions

ALGORITHM:

You import the necessary libraries You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, and perform some initial data exploration You group the data by date and resample it to a monthly frequency (beginning of the month You plot the time series data You import the necessary 'statsmodels' libraries for time series analysis You decompose the time series data into its additive components and plot them: You calculate the root mean squared error (RMSE) to evaluate the model's performance You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt- Winters model to the entire dataset and make future predictions You plot the original sales data and the predictions

PROGRAM :
~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Step 1: Load dataset
data = pd.read_csv('Tomato.csv')

# Convert Date column to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.set_index('Date')

# Choose target column (Average price)
target_col = 'Average'

# Step 2: Perform Augmented Dickey-Fuller test
result = adfuller(data[target_col].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Step 3: Train-test split (80% train, 20% test)
x = int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]

# Step 4: Fit AutoRegressive model
lag_order = 13
model = AutoReg(train_data[target_col], lags=lag_order)
model_fit = model.fit()

# Step 5: Plot ACF and PACF
plt.figure(figsize=(10, 6))
plot_acf(data[target_col], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(data[target_col], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Step 6: Predictions
predictions = model_fit.predict(start=len(train_data),
                                end=len(train_data)+len(test_data)-1)

# Step 7: Compare with test data
mse = mean_squared_error(test_data[target_col], predictions)
print('Mean Squared Error (MSE):', mse)

# Step 8: Plot results
plt.figure(figsize=(12, 6))
plt.plot(test_data[target_col], label='Test Data - Average Tomato Price')
plt.plot(predictions, label='Predictions - AR Model', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Average Tomato Price')
plt.title('AR Model Predictions vs Test Data (Tomato Dataset)')
plt.legend()
plt.grid()
plt.show()
~~~

OUTPUT:
<img width="716" height="540" alt="image" src="https://github.com/user-attachments/assets/6b0d6cf5-1fda-4b46-aab2-404b7a28e7b5" />
<img width="707" height="541" alt="image" src="https://github.com/user-attachments/assets/e523594d-ac9b-45fa-a261-5f682787597a" />
<img width="1257" height="680" alt="image" src="https://github.com/user-attachments/assets/0c38be8d-8bda-444a-879f-e8f45a7fc6b4" />


RESULT:
Thus the program run successfully based on the Holt Winters Method model.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the sales data
sales = pd.read_csv('sales.csv')

# Convert SaleDate to datetime format and extract relevant features
sales['SaleDate'] = pd.to_datetime(sales['SaleDate'])
sales['Year'] = sales['SaleDate'].dt.year
sales['Month'] = sales['SaleDate'].dt.month
sales['Day'] = sales['SaleDate'].dt.day

# Group by date and sum the quantities sold
sales_by_date = sales.groupby(['Year', 'Month', 'Day'])['Quantity'].sum().reset_index()

# Feature and target variables
X = sales_by_date[['Year', 'Month', 'Day']]
y = sales_by_date['Quantity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Root Mean Squared Error: {rmse}')

# Predict future sales (example: for a future date)
future_date = pd.DataFrame({'Year': [2024], 'Month': [9], 'Day': [1]})
future_sales_prediction = model.predict(future_date)

print(f'Predicted sales for {future_date.iloc[0]["Year"]}-{future_date.iloc[0]["Month"]}-{future_date.iloc[0]["Day"]}: {future_sales_prediction[0]}')

# Plot actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Sales')
plt.plot(y_pred, label='Predicted Sales', linestyle='--')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Test Data Index')
plt.ylabel('Quantity Sold')
plt.legend()
plt.show()

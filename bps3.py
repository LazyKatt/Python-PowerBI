import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the sales data
sales = pd.read_csv('sales.csv')

# Convert SaleDate to datetime format
sales['SaleDate'] = pd.to_datetime(sales['SaleDate'])

# Extract features: year, month, and day from SaleDate
sales['Year'] = sales['SaleDate'].dt.year
sales['Month'] = sales['SaleDate'].dt.month
sales['Day'] = sales['SaleDate'].dt.day

# Aggregate sales data by date (summing up the quantity sold each day)
sales_by_date = sales.groupby(['Year', 'Month', 'Day'])['Quantity'].sum().reset_index()

# Feature matrix X (using year, month, and day as features)
X = sales_by_date[['Year', 'Month', 'Day']]

# Target vector y (the total quantity sold)
y = sales_by_date['Quantity']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model by calculating the Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# Predict future sales for a specific future date (e.g., September 1, 2024)
future_date = pd.DataFrame({'Year': [2024], 'Month': [9], 'Day': [1]})
future_sales_prediction = model.predict(future_date)
print(f'Predicted sales for {future_date.iloc[0]["Year"]}-{future_date.iloc[0]["Month"]}-{future_date.iloc[0]["Day"]}: {future_sales_prediction[0]}')

# Plot the actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Sales', marker='o')
plt.plot(y_pred, label='Predicted Sales', marker='x', linestyle='--')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Test Data Index')
plt.ylabel('Quantity Sold')
plt.legend()
plt.show()

# Plot future sales prediction (example with historical data)
plt.figure(figsize=(10, 6))
plt.plot(sales_by_date.index, model.predict(X), label='Predicted Sales (Historical)', color='blue')
plt.scatter(len(sales_by_date), future_sales_prediction, color='red', label='Predicted Sales (Future)')
plt.title('Sales Prediction using Linear Regression')
plt.xlabel('Time Index')
plt.ylabel('Quantity Sold')
plt.legend()
plt.show()

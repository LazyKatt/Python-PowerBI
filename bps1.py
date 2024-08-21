import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV files
sales = pd.read_csv('sales.csv')
product_group = pd.read_csv('product_group.csv')
website_access = pd.read_csv('website_access.csv')

# Convert SaleDate to datetime format
sales['SaleDate'] = pd.to_datetime(sales['SaleDate'])

# Handle Missing Data

# Check for missing values
print(sales.isna().sum())
print(product_group.isna().sum())
print(website_access.isna().sum())

# Fill missing values in sales data (if any)
sales.fillna({'Quantity': sales['Quantity'].mean(), 'TotalAmount': sales['TotalAmount'].mean()}, inplace=True)

# Drop rows with missing values in product_group and website_access data
product_group.dropna(inplace=True)
website_access.dropna(inplace=True)

# Remove Duplicates

# Remove duplicates in all datasets
sales.drop_duplicates(inplace=True)
product_group.drop_duplicates(inplace=True)
website_access.drop_duplicates(inplace=True)

# Handle Outliers

# Identify outliers in the TotalAmount column using IQR
Q1 = sales['TotalAmount'].quantile(0.25)
Q3 = sales['TotalAmount'].quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
sales = sales[~((sales['TotalAmount'] < (Q1 - 1.5 * IQR)) | (sales['TotalAmount'] > (Q3 + 1.5 * IQR)))]

# Data Merging and Transformation

# Merge sales with product group data
sales_product_group = sales.merge(product_group, left_on='ProductDetailID', right_on='ProductGroupID')

# Optional: Create a new column for revenue per unit
sales['RevenuePerUnit'] = sales['TotalAmount'] / sales['Quantity']

# Visualizations

# Line Chart: Sales Over Time
sales_by_date = sales.groupby(sales['SaleDate'].dt.date)['Quantity'].sum()

plt.figure(figsize=(10, 6))
plt.plot(sales_by_date.index, sales_by_date.values, marker='o', linestyle='-', color='b')
plt.title('Total Products Sold Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar Chart: Revenue by Product Group
sales_by_group = sales_product_group.groupby('GroupName')['TotalAmount'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sales_by_group.plot(kind='bar', color='orange')
plt.title('Revenue by Product Group')
plt.xlabel('Product Group')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie Chart: Website Access by User Type
access_by_type = website_access['AccessType'].value_counts()

plt.figure(figsize=(8, 8))
access_by_type.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'salmon'])
plt.title('Website Access Distribution by User Type')
plt.ylabel('')
plt.tight_layout()
plt.show()

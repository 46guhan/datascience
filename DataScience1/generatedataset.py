""" import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)

# Sample size
n = 200

# Departments
departments = ['HR', 'Sales', 'Engineering', 'Finance', 'Marketing']

# Create the dataset
data = {
    'EmployeeID': range(1001, 1001 + n),
    'Age': np.random.randint(22, 60, size=n),
    'Department': np.random.choice(departments, size=n),
    'EducationLevel': np.random.randint(1, 5, size=n),  # 1=HighSchool, 4=PhD
    'YearsAtCompany': np.random.randint(0, 20, size=n),
    'JobSatisfaction': np.random.randint(1, 6, size=n),  # 1-5 scale
    'Salary': np.random.randint(30000, 120000, size=n),
    'WorkLifeBalance': np.random.randint(1, 6, size=n),  # 1-5 scale
}

# Generate Attrition based on some logic
df = pd.DataFrame(data)
df['Attrition'] = df.apply(lambda x: 1 if (x['JobSatisfaction'] < 3 and x['WorkLifeBalance'] < 3 and x['YearsAtCompany'] < 3) else 0, axis=1)

# Optional: Convert Attrition to Yes/No
df['Attrition'] = df['Attrition'].map({1: 'Yes', 0: 'No'})

# Preview
print(df.head())
df.to_csv("dataset/employee_supervised_dataset.csv", index=False)
 """
""" 
import pandas as pd
import numpy as np

# Set seed
np.random.seed(7)

# Sample size
n = 150

# Departments
departments = ['HR', 'Sales', 'Engineering', 'Finance', 'Marketing']

# Create data
data = {
    'EmployeeID': range(2001, 2001 + n),
    'Age': np.random.randint(22, 60, size=n),
    'Department': np.random.choice(departments, size=n),
    'EducationLevel': np.random.randint(1, 5, size=n),
    'YearsAtCompany': np.random.randint(0, 20, size=n),
    'JobSatisfaction': np.random.randint(1, 6, size=n),
    'Salary': np.random.randint(30000, 120000, size=n),
    'WorkLifeBalance': np.random.randint(1, 6, size=n)
}

df_unsupervised = pd.DataFrame(data)

# Preview
print(df_unsupervised.head())
df_unsupervised.to_csv("dataset/employee_unsupervised_dataset.csv", index=False) """

""" 
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate 1000 hourly timestamps starting from 2025-01-01
timestamps = pd.date_range(start='2025-01-01 00:00:00', periods=1000, freq='H')

# Generate synthetic temperature, humidity, and power consumption
temperature = np.round(20 + 5 * np.sin(np.linspace(0, 50, 1000)) + np.random.normal(0, 1, 1000), 2)
humidity = np.round(50 + 10 * np.cos(np.linspace(0, 20, 1000)) + np.random.normal(0, 2, 1000), 2)
power_consumption = np.round(1.5 + 0.5 * np.sin(np.linspace(0, 30, 1000)) + np.random.normal(0, 0.1, 1000), 2)

# Create a DataFrame
df = pd.DataFrame({
    'Timestamp': timestamps,
    'Temperature (°C)': temperature,
    'Humidity (%)': humidity,
    'Power Consumption (kWh)': power_consumption
})

# Save to CSV
df.to_csv('dataset/time_series_data.csv', index=False)

print("CSV file 'time_series_data.csv' created with 1000 rows.")
 """
""" 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range
date_range = pd.date_range(start='2018-01-01', periods=72, freq='M')  # 6 years of monthly data

# Create synthetic data with trend + seasonality + noise
trend = np.linspace(100, 300, 72)  # upward trend
seasonality = 20 * np.sin(np.linspace(0, 3 * np.pi, 72))  # seasonal effect
noise = np.random.normal(loc=0, scale=10, size=72)  # random noise

# Combine all components
sales = trend + seasonality + noise

# Create DataFrame
df = pd.DataFrame({
    'Date': date_range,
    'Sales': np.round(sales, 2)
})

# Set date as index (optional for time series)
df.set_index('Date', inplace=True)

# Show first few rows
print(df.head())

# Plot the data
df.plot(figsize=(10, 4), title='Synthetic Monthly Sales Data')
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# Optionally save to CSV
df.to_csv('forecasting_dataset.csv')
 """
""" 
from sklearn.datasets import make_classification
import pandas as pd

# Generate synthetic dataset
X, y = make_classification(
    n_samples=500,        # Number of data points
    n_features=10,        # Total features (dimensions)
    n_informative=5,      # Informative features
    n_redundant=2,        # Redundant (correlated) features
    n_classes=2,          # Binary classification
    random_state=42
)

# Convert to DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y

# Save to CSV (optional)
df.to_csv("dim_reduction_dataset.csv", index=False)

print(df.head())
 """

"""
import random
import pandas as pd

# List of items in the supermarket
items = ['milk', 'bread', 'butter', 'eggs', 'cheese', 'apples', 'banana', 'juice', 'chips', 'yogurt', 'coffee', 'tea']

# Generate synthetic transactions
num_transactions = 1000
transactions = []

for tid in range(1, num_transactions + 1):
    num_items = random.randint(1, 6)  # Each transaction has 1 to 6 items
    selected_items = random.sample(items, num_items)
    item_str = ",".join(selected_items)
    transactions.append([tid, item_str])

# Create DataFrame
df = pd.DataFrame(transactions, columns=['TransactionID', 'Items'])

# Save to CSV
df.to_csv("dataset/supermarket_transactions.csv", index=False)
"""


import pandas as pd
import json
"""
# Sample data to write to the files
data = {
    "Name": ["John", "Alice", "Bob"],
    "Age": [28, 24, 22],
    "City": ["New York", "Los Angeles", "Chicago"]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Write to Text File
with open('data.txt', 'w') as file:
    for row in df.values:
        file.write(" | ".join(map(str, row)) + "\n")
    print("Data written to text file (data.txt)")

# Write to CSV File
df.to_csv('data.csv', index=False)
print("Data written to CSV file (data.csv)")

# Write to Tab-Delimited File
df.to_csv('data_tab.txt', sep='\t', index=False)
print("Data written to Tab-delimited file (data_tab.txt)")

# Write to JSON File
df.to_json('data.json', orient='records', lines=True)
print("Data written to JSON file (data.json)")
"""
# Read from Text File
with open('data.txt', 'r') as file:
    text_data = file.readlines()
    print("\nData read from text file (data.txt):")
    print(text_data)

# Read from CSV File
csv_data = pd.read_csv('data.csv')
print("\nData read from CSV file (data.csv):")
print(csv_data)

# Read from Tab-Delimited File
tab_data = pd.read_csv('data_tab.txt', sep='\t')
print("\nData read from Tab-delimited file (data_tab.txt):")
print(tab_data)

# Read from JSON File
# Read from JSON File (one record per line)
with open('data.json', 'r') as file:
    json_data = [json.loads(line) for line in file]
    print("\nData read from JSON file (data.json):")
    print(json_data)


print(" CSV file 'supermarket_transactions.csv' generated successfully!")

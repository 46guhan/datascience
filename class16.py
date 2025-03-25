
import pandas as pd
import matplotlib.pyplot as plt
import pymysql as mysql
# Connect to the MySQL database
conn = mysql.connect(host="localhost",user="root",password="root",database="sales1")

# SQL query to retrieve sales data
query = "SELECT * FROM Sales"

# Fetch data into a pandas DataFrame
sales_df = pd.read_sql(query, conn)

# Close the database connection
conn.close()

# Display the first few rows of the DataFrame
print(sales_df.head())

# Perform data analysis or visualization
# For example, let's visualize the distribution of sales quantities
plt.hist(sales_df['quantity'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Sales Quantities')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.show()

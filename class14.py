#data science
""" import pandas as pd
df = pd.read_csv("dataset/Tennisdataset1.2.csv")
print(df.head(5)) """

#dataEngineering
""" import pymysql as mysql
connection = mysql.connect(host="localhost",user="root",password="root",database="javabank")
cursor=connection.cursor()
cursor.execute("show tables")
for x in cursor:
   print(x)  """
 

#data source

import csv
import json
import pandas as pd

# Writing to a text file
with open('example.txt', 'w') as f:
    f.write("Hello, world!\n")
    f.write("This is a text file.")

# Reading from a text file
with open('example.txt', 'r') as f:
    text_data = f.read()
    print("Text File Content:")
    print(text_data) 


# Writing to a CSV file
csv_data = [
    ['Name', 'Age', 'Country'],
    ['John', 30, 'USA'],
    ['Alice', 25, 'UK'],
    ['Bob', 35, 'Canada']
]

with open('example.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

# Reading from a CSV file
with open('example.csv', 'r') as f:
    reader = csv.reader(f)
    csv_data_read = list(reader)
    print("\nCSV File Content:")
    for row in csv_data_read:
        print(row)



# Writing to a JSON file
json_data = {
    "Name": "John",
    "Age": 30,
    "Country": "USA"
}

with open('example.json', 'w') as f:
    json.dump(json_data, f)

# Reading from a JSON file
with open('example.json', 'r') as f:
    json_data_read = json.load(f)
    print("\nJSON File Content:")
    print(json_data_read) 

# Writing to an Excel file using Pandas
excel_data = pd.DataFrame({
    'Name': ['John', 'Alice', 'Bob'],
    'Age': [30, 25, 35],
    'Country': ['USA', 'UK', 'Canada']
})

excel_data.to_excel('example.xlsx', index=False)

# Reading from an Excel file using Pandas
excel_data_read = pd.read_excel('example.xlsx')
print("\nExcel File Content:")
print(excel_data_read)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

""" # Parameters
mu = 0  # Mean
sigma = 1  # Standard deviation

# Calculate PDF and CDF for a specific value
x = 1.5
pdf_value = norm.pdf(x, mu, sigma)#probability density function
cdf_value = norm.cdf(x, mu, sigma)#cumulative distribution function

print("PDF at x =", x, ":", pdf_value)
print("CDF at x =", x, ":", cdf_value)

# Parameters
mu = 0  # Mean
sigma = 1  # Standard deviation
num_samples = 1000  # Number of samples

# Generate random numbers from a normal distribution
samples = np.random.normal(mu, sigma, num_samples)

# Plot histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g')

# Plot the PDF (Probability Density Function)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram and PDF of a Normal Distribution')
plt.show() """
""" 

import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
# Step 1: Load the dataset from CSV
data = pd.read_csv('dataset\\adult.csv')

# Step 2: Extract the column containing the data you want to analyze
# For example, if your CSV file has a column named 'values', you can do:
values = data['age']

# Step 3: Fit a normal distribution to the data or estimate its parameters
mu, sigma = norm.fit(values)

# Step 4: Optionally, visualize the original data and the fitted normal distribution
plt.hist(values, bins=30, density=True, alpha=0.6, color='g')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)

plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram and Fitted Normal Distribution')
plt.show()

# Print mean and standard deviation of the fitted normal distribution
print("Mean:", mu)
print("Standard Deviation:", sigma) """

#z-score
""" 
import numpy as np

# Define your dataset
data = np.array([10, 12, 15, 18, 20, 22, 25, 28, 30])

# Calculate the mean and standard deviation of the dataset
mean = np.mean(data)
std_dev = np.std(data)

# Define the value for which you want to calculate the z-score
value = 22

# Calculate the z-score
z_score = (value - mean) / std_dev

print("Z-score:", z_score) """


""" import pandas as pd
from scipy.stats import zscore
# Step 1: Load the dataset from CSV
data = pd.read_csv('dataset/adult.csv')
data["zscore"]=zscore(data["age"])
print(data)  """


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate some sample data
np.random.seed(42)
data = np.random.normal(loc=10, scale=2, size=100)  # Normal distribution with mean=10 and std=2
print(data)
# Calculate confidence interval
confidence_level = 0.95  # 95% confidence level
mean = np.mean(data)
print(mean)
std_err = stats.sem(data)
print(std_err)
confidence_interval = stats.norm.interval(confidence_level, loc=mean, scale=std_err)
print(confidence_interval)

# Plot the data and confidence interval
plt.hist(data, bins=20, alpha=0.5, color='skyblue', edgecolor='black')
plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(confidence_interval[0], color='green', linestyle='dashed', linewidth=1, label='Confidence Interval')
plt.axvline(confidence_interval[1], color='green', linestyle='dashed', linewidth=1)
plt.legend()
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Sample Data with 95% Confidence Interval')
# plt.show()



""" import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data for Z-distribution
mu = 0  # mean
sigma = 1  # standard deviation
x = np.linspace(-5, 5, 1000)  # range of x values

z_pdf = norm.pdf(x, mu, sigma)  # Z-distribution PDF
print(z_pdf)
# Plotting the Z-distribution
plt.plot(x, z_pdf, label='Z-distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Standard Normal (Z) Distribution')
plt.legend()
plt.grid(True)
plt.show() """

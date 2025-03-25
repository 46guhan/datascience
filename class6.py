import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# Parameters for the t-distribution
df = 5  # degrees of freedom
x = np.linspace(-5, 5, 1000)  # range of x values
t_pdf = t.pdf(x, df)  # t-distribution PDF

# Plotting the t-distribution
plt.plot(x, t_pdf, label=f't-distribution (df={df})')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title(f't-distribution with {df} degrees of freedom')
plt.legend()
plt.grid(True)
plt.show()

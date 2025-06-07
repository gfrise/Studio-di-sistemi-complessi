#processo OU con dpf e AC
import numpy as np
import matplotlib.pyplot as plt

#Eulero
dt, n, gamma = 0.01, 10**3, 1.0
x = np.zeros(n)
for i in range (1,n):
    x[i] = x[i-1] - x[i-1]*dt + 0.5*np.sqrt(dt)*np.random.randn()

#PDF
h, e = np.histogram(x, bins=50, density=True)
c = 0.5 * (e[1:]+e[:-1])
plt.plot(c,h)
plt.title("PDF")
plt.show()

# Autocorrelazione semplice
mean = np.mean(x)
var = np.var(x)
max_lag = n // 10
acf = np.zeros(max_lag)
for lag in range(max_lag):
    acf[lag] = np.sum((x[:n-lag] - mean) * (x[lag:] - mean)) / ((n - lag) * var)

plt.plot(acf)
plt.title("Autocorrelazione")
plt.show()
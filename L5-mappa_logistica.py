import numpy as np
import matplotlib.pyplot as plt

def logistic(r, x):
    return r * x * (1 - x)

r_min, r_max, r_step, x0, NN = 1.0, 4.0, 0.001, 0.1, 100
r_values, R, X = np.arange(r_min + r_step, r_max, r_step), [], []

rates = np.linspace(1, 4, 2000) #crea 2k numeri tra 1 e 4
burn_in = 1000
samples = 100
x0 = 0.5
for rate in rates:
    x = x0
    for _ in range(burn_in):
        x = logistic(rate, x)
    for _ in range(samples):
        x = logistic(rate, x)
        R.append(rate)
        X.append(x)

# for r in r_values:
#     x = x0
#     # Iterazioni della mappa logistica
#     for t in range(NN):
#         x = logistic(r,x)
#         R.append(r)
#         X.append(x)

plt.figure(figsize=(10, 6))
plt.plot(R, X, ',', alpha=0.3, color='black')
plt.xlabel("r")
plt.ylabel("x")
plt.title("Diagramma delle Biforcazioni ")
plt.show()
plt.figure(figsize=(8,8))
plt.plot(R, X, ',', alpha=0.4, color='black')
plt.xlabel('r')
plt.ylabel('x')
plt.title('Diagramma delle Biforcazioni (Zoom)')
plt.xlim(3.4, 3.6)
plt.ylim(0.4, 0.7)
plt.show()
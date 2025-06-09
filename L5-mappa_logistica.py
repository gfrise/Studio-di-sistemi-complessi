# import numpy as np
# import matplotlib.pyplot as plt

# def logistic(r, x):
#     return r * x * (1 - x)

# rates = np.linspace(1, 4, 20000) #crea 2k numeri tra 1 e 4
# burn_in = 1000
# samples = 100
# x0 = 0.5

# R, X = [], []
# for rate in rates:
#     x = x0
#     for _ in range(burn_in):
#         x = logistic(rate, x)
#     for _ in range(samples):
#         x = logistic(rate, x)
#         R.append(rate)
#         X.append(x)

# plt.figure(figsize=(8,8))
# plt.plot(R, X, ',', alpha=0.4)
# plt.xlabel('r')
# plt.ylabel('x')
# plt.show()

# plt.figure(figsize=(8,8))
# plt.plot(R, X, ',', alpha=0.4)
# plt.xlabel('r')
# plt.ylabel('x')
# plt.title('Diagramma delle Biforcazioni (Zoom)')

# # Zoom su un intervallo specifico
# plt.xlim(3.4, 3.6)
# plt.ylim(0.4, 0.7)

# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Parametri
r_min = 1.0
r_max = 4.0
r_step = 0.001
x0 = 0.1
NN = 1000  # Iterazioni per ogni valore di r

# Costruzione della lista di r
r_values = np.arange(r_min + r_step, r_max, r_step)

# Liste dove salvare i risultati
R = []
X = []

# Loop su tutti i r
for r in r_values:
    x = x0
    # Iterazioni della mappa logistica
    for t in range(NN):
        x = r * x * (1 - x)
        R.append(r)
        X.append(x)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(R, X, ',', alpha=0.3, color='black')
plt.xlabel("r")
plt.ylabel("x")
plt.title("Diagramma delle Biforcazioni - Tradotto da C a Python")
plt.show()

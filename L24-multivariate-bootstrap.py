#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Impostazioni principali
np.random.seed(42)
n_points = 1000    # Numero di punti nel dataset originale
n_boot = 500       # Numero di repliche bootstrap

# Generazione di dati bivariati normali con correlazione
mu = [0, 0]
sigma = [[1.0, 0.8], [0.8, 1.0]]
X = np.random.multivariate_normal(mu, sigma, n_points)

# Arrays per salvare stime da ogni campione bootstrap
means = np.zeros((n_boot, 2))       # medie [x̄, ȳ]
covariances = np.zeros((n_boot, 2, 2))  # matrici di covarianza

# Bootstrap: estrazioni con rimpiazzo
for i in range(n_boot):
    sample = X[np.random.randint(0, n_points, size=n_points)]
    means[i] = np.mean(sample, axis=0)
    covariances[i] = np.cov(sample.T)

# Plot 1: distribuzione delle medie campionarie (scatter)
plt.figure(figsize=(6,6))
plt.scatter(means[:, 0], means[:, 1], alpha=0.5, s=15)
plt.xlabel("Media X")
plt.ylabel("Media Y")
plt.title("Distribuzione bootstrap delle medie")
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

# Plot 2: istogramma della varianza X
plt.figure()
plt.hist(covariances[:, 0, 0], bins=30, alpha=0.7, color='skyblue', edgecolor='k')
plt.xlabel("Varianza X")
plt.title("Distribuzione bootstrap: Varianza X")
plt.tight_layout()

# Plot 3: istogramma della covarianza XY
plt.figure()
plt.hist(covariances[:, 0, 1], bins=30, alpha=0.7, color='lightcoral', edgecolor='k')
plt.xlabel("Covarianza XY")
plt.title("Distribuzione bootstrap: Covarianza")
plt.tight_layout()

# Plot 4: istogramma della varianza Y
plt.figure()
plt.hist(covariances[:, 1, 1], bins=30, alpha=0.7, color='palegreen', edgecolor='k')
plt.xlabel("Varianza Y")
plt.title("Distribuzione bootstrap: Varianza Y")
plt.tight_layout()

# Mostra tutti i grafici
plt.show()

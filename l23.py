#!/usr/bin/env python3
"""
Script per simulare:
1. Processo di Ornstein-Uhlenbeck con shuffling che preserva la PDF e autocorrelazione
2. Processo di Wiener con PDF e varianza

Tutti i parametri sono definiti nel codice.
"""
import numpy as np

# Parametri processo OU
gamma = 0.5
enne = 100
x0 = 0.0
nR = 1000
nbin = 50
tmax = 200
seed_ou = 42
seed_shuffle = 123

# Parametri processo Wiener
seed_wiener = 24

# Funzioni utili
def compute_pdf(x, bins):
    pdf, edges = np.histogram(x, bins=bins, density=True)
    return edges[:-1], pdf

def autocorrelation(x, tmax):
    x -= x.mean()
    result = np.correlate(x, x, mode='full')
    acf = result[result.size // 2:] / result[result.size // 2]
    return acf[:tmax]

# Simula OU
dt = 1.0 / enne
N = nR * enne
rng_ou = np.random.default_rng(seed_ou)
Z_ou = rng_ou.normal(0, np.sqrt(2 * dt), N)
X_ou = np.zeros(N)
X_ou[0] = x0
for i in range(1, N):
    X_ou[i] = X_ou[i-1] - gamma * X_ou[i-1] * dt + Z_ou[i]

# Shuffling
X_shuff = X_ou.copy()
np.random.default_rng(seed_shuffle).shuffle(X_shuff)

# Output PDF e autocorrelazione OU
edges_ou, pdf_ou = compute_pdf(X_shuff, nbin)
acf_ou = autocorrelation(X_shuff, tmax)

# Simula Wiener
Z_w = np.random.default_rng(seed_wiener).normal(0, np.sqrt(dt), N)
X_wiener = np.zeros(N)
X_wiener[0] = x0
for i in range(1, N):
    X_wiener[i] = X_wiener[i-1] + Z_w[i]

edges_w, pdf_w = compute_pdf(X_wiener, nbin)
Xw_matrix = X_wiener.reshape(nR, enne)
var_wiener = Xw_matrix.var(axis=0)

# Salva risultati
np.savetxt('serie_ou.dat', X_ou)
np.savetxt('serie_ou_shuff.dat', X_shuff)
np.savetxt('pdf_ou.dat', np.column_stack((edges_ou, pdf_ou)))
np.savetxt('auto_corr_ou.dat', np.column_stack((np.arange(len(acf_ou)), acf_ou)))
np.savetxt('serie_wiener.dat', X_wiener)
np.savetxt('pdf_wiener.dat', np.column_stack((edges_w, pdf_w)))
np.savetxt('variance_wiener.dat', var_wiener)

print("Salvati i file per OU e Wiener.")

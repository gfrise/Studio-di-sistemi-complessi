#!/usr/bin/env python3
"""
Simulazione di un processo Ornstein-Uhlenbeck con calcolo del secondo momento di ciascuna traiettoria e statistiche sull'ensemble.
Legge i parametri da 'input_parametersOUmom.dat', genera 'nR' punti casuali per traiettoria,
calcola per ogni traiettoria il momento centrale di ordine 2 e poi i momenti (1°–6°) sull'insieme dei secondi momenti.
"""
import sys
import numpy as np

# Calcola il momento centrale di ordine n
def central_moment(data, n):
    mu = np.mean(data)
    return np.mean((data - mu) ** n)

# Parsing di nR da linea di comando
def parse_args():
    if len(sys.argv) != 2:
        print(f"Uso: {sys.argv[0]} <nR>")
        sys.exit(1)
    try:
        return int(sys.argv[1])
    except ValueError:
        print("Errore: nR deve essere un intero.")
        sys.exit(1)

# Parametri
gammaOU, enne, nENS = np.loadtxt('input_parametersOUmom.dat')
enne = int(enne)
nENS = int(nENS)
delt = 1.0 / enne
staz = 1
nR = parse_args()
total_steps = staz * nR * enne

# Array per i secondi momenti di ogni traiettoria
second_moments = np.zeros(nENS)

for k in range(nENS):
    # Rumore gaussiano
    Z = np.random.normal(0.0, np.sqrt(2 * delt), total_steps)

    # Simulazione OU
    X = np.zeros(total_steps)
    X[0] = np.random.rand() * 2 - 1
    for i in range(1, total_steps):
        X[i] = X[i-1] + (-gammaOU * X[i-1] * delt + Z[i])

    # Downsample e momento 2
    X_samp = X[::enne][:nR]
    second_moments[k] = central_moment(X_samp, 2)

# Calcolo momenti sull'ensemble dei secondi momenti
ensemble_moments = [np.mean(second_moments)] + [central_moment(second_moments, m) for m in range(2, 7)]

# Output
with open('momenti2.dat', 'w') as f:
    f.write(f"{nR} " + " ".join(f"{val:.6f}" for val in ensemble_moments) + "\n")

print("Simulazione completata. Risultati in 'momenti2.dat'.")

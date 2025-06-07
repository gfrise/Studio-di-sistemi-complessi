#!/usr/bin/env python3
"""
Simulazione di un processo Ornstein-Uhlenbeck e calcolo dei momenti (1°–6°).
Legge i parametri da 'input_parametersOUmom.dat' e accetta 'nR' come argomento.
Salva i risultati in 'momenti.dat'.
"""
import sys
import numpy as np

# Funzione per momento centrale di ordine n
def central_moment(data, n):
    mu = np.mean(data)
    return np.mean((data - mu)**n)

# Verifica argomenti da riga di comando
def parse_args():
    if len(sys.argv) != 2:
        print(f"Uso: {sys.argv[0]} <nR>")
        sys.exit(1)
    try:
        return int(sys.argv[1])
    except ValueError:
        print("Errore: nR deve essere un intero.")
        sys.exit(1)

# Lettura dei parametri da file
# Il file deve contenere tre valori: gammaOU, enne, nENS
params = np.loadtxt('input_parametersOUmom.dat')
gammaOU = params[0]
enne = int(params[1])
nENS = int(params[2])

delt = 1.0 / enne
staz = 1

# Estrae nR da riga di comando
nR = parse_args()

# Array per memorizzare le medie delle traiettorie
means = np.zeros(nENS)

# Loop sull'ensemble
total_steps = staz * nR * enne
for iter_idx in range(nENS):
    # Generazione rumore gaussiano con NumPy
    Z1 = np.random.normal(loc=0.0, scale=np.sqrt(2 * delt), size=total_steps)

    # Simulazione processo OU
    Xs = np.zeros(total_steps)
    Xs[0] = np.random.rand() * 2.0 - 1.0  # punto iniziale in [-1,1]
    for i in range(1, total_steps):
        dX = -gammaOU * Xs[i-1] * delt + Z1[i]
        Xs[i] = Xs[i-1] + dX

    # Sottocampionamento ogni 'enne' step
    X = Xs[::enne][:nR]
    means[iter_idx] = np.mean(X)

# Calcolo dei momenti sull'ensemble delle medie
momenti = [np.mean(means)] + [central_moment(means, k) for k in range(2,7)]

# Scrittura risultati su file
with open('momenti.dat', 'w') as f:
    f.write(f"{nR} " + " ".join(f"{m:.6f}" for m in momenti) + "\n")

print("Simulazione completata. Risultati in 'momenti.dat'.")

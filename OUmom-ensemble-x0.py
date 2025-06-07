#!/usr/bin/env python3
"""
Simulazione di un processo Ornstein-Uhlenbeck e calcolo dei momenti (1°–6°).
Legge i parametri da 'input_parametersOUmom.dat', i punti iniziali da 'serienumericaOU.dat',
accetta 'nR' come argomento da riga di comando, e scrive i risultati in 'momenti.dat'.
"""
import sys
import numpy as np

# Funzione per momento centrale di ordine n
def central_moment(data, n):
    mu = np.mean(data)
    return np.mean((data - mu) ** n)

# Verifica e parsing degli argomenti da linea di comando
def parse_args():
    if len(sys.argv) != 2:
        print(f"Uso: {sys.argv[0]} <nR>")
        sys.exit(1)
    try:
        return int(sys.argv[1])
    except ValueError:
        print("Errore: nR deve essere un intero.")
        sys.exit(1)

# --- Main ---
nR = parse_args()

# Lettura parametri: gammaOU, enne (punti/unità di tempo), nENS (size ensemble)
gammaOU, enne, nENS = np.loadtxt("input_parametersOUmom.dat")

enne = int(enne)
nENS = int(nENS)
delt = 1.0 / enne
staz = 1
total_steps = staz * nR * enne

# Lettura dei punti iniziali
try:
    x0_array = np.loadtxt("serienumericaOU.dat")
except Exception as e:
    print(f"Errore nel leggere 'serienumericaOU.dat': {e}")
    sys.exit(1)

if len(x0_array) != nENS:
    print("Errore: il numero di punti iniziali non corrisponde a nENS.")
    sys.exit(1)

means = np.zeros(nENS)

# Simulazioni per ogni membro dell'ensemble
for iter_idx in range(nENS):
    x0 = x0_array[iter_idx]

    # Genera rumore gaussiano direttamente
    Z1 = np.random.normal(loc=0.0, scale=np.sqrt(2 * delt), size=total_steps)

    # Simula il processo OU
    Xs = np.zeros(total_steps)
    Xs[0] = x0
    for i in range(1, total_steps):
        dX = -gammaOU * Xs[i - 1] * delt + Z1[i]
        Xs[i] = Xs[i - 1] + dX

    # Sottocampionamento
    X = Xs[::enne][:nR]
    means[iter_idx] = np.mean(X)

# Calcolo dei momenti sull'ensemble delle medie
momenti = [np.mean(means)] + [central_moment(means, k) for k in range(2, 7)]

# Scrittura risultati
with open("momenti.dat", "w") as f:
    f.write(f"{nR} " + " ".join(f"{m:.6f}" for m in momenti) + "\n")

print("Simulazione completata. Risultati in 'momenti.dat'.")

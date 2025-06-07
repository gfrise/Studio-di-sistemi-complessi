#!/usr/bin/env python3
import sys
import numpy as np

def central_moment(data, n):
    mu = data.mean()
    return ((data - mu)**n).mean()

# --- Parametri e parsing ---
if len(sys.argv) != 2:
    print(f"Uso: {sys.argv[0]} <nR>"); sys.exit(1)
nR = int(sys.argv[1])

gamma, enne_f, nENS_f = np.loadtxt('input_parametersOUmom.dat')
enne, nENS = int(enne_f), int(nENS_f)
delt = 1.0 / enne
total_steps = nR * enne

#--- Simulazione e raccolta delle medie per traiettoria ---
means = np.empty(nENS)
for k in range(nENS):
    # rumore gaussiano
    Z = np.random.normal(0, np.sqrt(2*delt), total_steps)
    X = np.empty(total_steps)
    X[0] = np.random.uniform(-1, 1)
    for i in range(1, total_steps):
        X[i] = X[i-1] + (-gamma * X[i-1] * delt + Z[i])
    # sottocampionamento (qui è già nR*enne, quindi prendo tutto)
    means[k] = X.reshape(nR, enne).mean(axis=1).mean()

#--- Momenti sull'ensemble delle medie ---
moments = [means.mean()] + [central_moment(means, m) for m in range(2,7)]

#--- Scrittura ---
with open('momenti.dat','w') as f:
    f.write(f"{nR} " + " ".join(f"{v:.6f}" for v in moments) + "\n")

#!/usr/bin/env python3
"""
Questo script Python integra le varie simulazioni e analisi richieste senza l'uso di file esterni.

Compito OU:
  - Processo Ornstein-Uhlenbeck via Langevin
  - Generazione rumore gaussiano
  - Shuffling preservando la PDF
  - Statistiche base, PDF e autocorrelazione

Compito RISK:
  - Processo "RISK" (drift ±kappa a seconda del segno)
  - Generazione rumore gaussiano
  - Shuffling, statistiche, PDF, autocorrelazione

Compito Wiener:
  - Moto Browniano (processo di Wiener)
  - PDF e varianza in funzione del tempo

Tutti i parametri iniziali (gamma, kappa, dt, x0, nR, nbin, tmax, seed) sono definiti in testa.
"""
import numpy as np
import matplotlib.pyplot as plt

# --- Parametri comuni ---
enne = 100        # punti/unità di tempo
dt = 1.0/enne
nR = 10000
nbin = 50
tmax = 200
x0 = 0.0

gamma_ou = 1.0
kappa = 0.5
seed_ou = 42
seed_risk = 43
seed_shuffle = 123
seed_wiener = 44

# Funzione autocorrelazione
def autocorr(x, maxlag):
    N = len(x)
    m = x.mean()
    sd = np.sqrt(((x-m)**2).mean())
    corr = []
    for t in range(maxlag):
        num = ((x[:N-t]-m)*(x[t:]-m)).sum()/(N-t)
        corr.append(num/(sd*sd))
    return np.array(corr)

def autocorrelation(x, tmax):
    x -= x.mean()
    result = np.correlate(x, x, mode='full')
    acf = result[result.size // 2:] / result[result.size // 2]
    return acf[:tmax]

# Compito OU
def simulate_ou():
    rng = np.random.default_rng(seed_ou)
    Z = rng.normal(0, np.sqrt(2*dt), nR*enne)
    X = np.empty(nR*enne)
    X[0] = x0
    for i in range(1, len(X)):
        X[i] = X[i-1] - gamma_ou*X[i-1]*dt + Z[i]
    Xs = X.reshape(nR, enne)[:, 0]
    Xs_sh = Xs.copy(); np.random.default_rng(seed_shuffle).shuffle(Xs_sh)
    pdf, bins = np.histogram(Xs_sh, bins=nbin, density=True)
    ac = autocorr(Xs_sh, tmax)
    stats = Xs_sh.mean(), Xs_sh.std(), Xs_sh.min(), Xs_sh.max()
    return Xs_sh, stats, pdf, bins, ac

# Compito RISK
def simulate_risk():
    rng = np.random.default_rng(seed_risk)
    Z = rng.normal(0, np.sqrt(2*dt), nR*enne)
    X = np.empty(nR*enne)
    X[0] = x0
    for i in range(1, len(X)):
        drift = kappa*dt if X[i-1] < 0 else -kappa*dt
        X[i] = X[i-1] + drift + Z[i]
    Xs = X.reshape(nR, enne)[:, 0]
    Xs_sh = Xs.copy(); np.random.default_rng(seed_shuffle).shuffle(Xs_sh)
    pdf, bins = np.histogram(Xs_sh, bins=nbin, density=True)
    ac = autocorr(Xs_sh, tmax)
    stats = Xs_sh.mean(), Xs_sh.std(), Xs_sh.min(), Xs_sh.max()
    return Xs_sh, stats, pdf, bins, ac

# Compito Wiener
def simulate_wiener():
    rng = np.random.default_rng(seed_wiener)
    Z = rng.normal(0, np.sqrt(dt), nR*enne)
    W = np.empty(nR*enne)
    W[0] = x0
    for i in range(1, len(W)):
        W[i] = W[i-1] + Z[i]
    Ws = W.reshape(nR, enne)[:, 0]
    pdf, bins = np.histogram(Ws, bins=nbin, density=True)
    var_t = W.reshape(nR, enne).var(axis=0)
    stats = Ws.mean(), Ws.std(), Ws.min(), Ws.max()
    return Ws, stats, pdf, bins, var_t

# Simula Wiener
Z_w = np.random.default_rng(seed_wiener).normal(0, np.sqrt(dt), N)
X_wiener = np.zeros(N)
X_wiener[0] = x0
for i in range(1, N):
    X_wiener[i] = X_wiener[i-1] + Z_w[i]

# Esecuzione e visualizzazione
def main():
    ou_data, ou_stats, ou_pdf, ou_bins, ou_ac = simulate_ou()
    risk_data, risk_stats, risk_pdf, risk_bins, risk_ac = simulate_risk()
    wien_data, wien_stats, wien_pdf, wien_bins, wien_var = simulate_wiener()

    print("--- OU ---")
    print("Media, Dev.Std, Min, Max:", ou_stats)
    print("--- RISK ---")
    print("Media, Dev.Std, Min, Max:", risk_stats)
    print("--- WIENER ---")
    print("Media, Dev.Std, Min, Max:", wien_stats)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs[0,0].hist(ou_data, bins=nbin, density=True, alpha=0.7)
    axs[0,0].set_title("OU PDF")
    axs[0,1].plot(ou_ac)
    axs[0,1].set_title("OU Autocorrelation")
    axs[0,2].plot(ou_data[:200])
    axs[0,2].set_title("OU Time Series")

    axs[1,0].hist(risk_data, bins=nbin, density=True, alpha=0.7)
    axs[1,0].set_title("RISK PDF")
    axs[1,1].plot(risk_ac)
    axs[1,1].set_title("RISK Autocorrelation")
    axs[1,2].plot(risk_data[:200])
    axs[1,2].set_title("RISK Time Series")

    axs[2,0].hist(wien_data, bins=nbin, density=True, alpha=0.7)
    axs[2,0].set_title("Wiener PDF")
    axs[2,1].plot(wien_var)
    axs[2,1].set_title("Wiener Variance(t)")
    axs[2,2].plot(wien_data[:200])
    axs[2,2].set_title("Wiener Time Series")

    plt.tight_layout()
    plt.show()
    print("Nota: OU usa drift proporzionale a X; RISK usa drift ±kappa; Wiener nessun drift.")

if __name__ == '__main__':
    main()

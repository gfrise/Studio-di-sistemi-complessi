#!/usr/bin/env python3
"""
LABORATORIO 04

Integrazione numerica di:
  - Ornstein–Uhlenbeck a singola scala temporale via simulazioni Ensemble-Average
  - Processo di Risken multiscala via simulazioni Ensemble-Average

PROGRAMMAZIONE
Compito 1: OU_EA_input.c
  Create un codice che simuli un processo di Ornstein–Uhlenbeck a partire dalla sua equazione
  di Langevin attraverso simulazioni Ensemble-Average. Il codice deve anche prevedere la
  costruzione della funzione densità di probabilità (area normalizzata ad 1) e della funzione
  di autocorrelazione. Fate in modo che i parametri rilevanti della simulazione siano inseriti
  dall’esterno, per esempio tramite lettura da file che contenga tutti i parametri.

Compito 2: RISK_EA_input.c
  Create un codice che simuli il processo multiscala definito dal coefficiente di drift
  non lineare attraverso simulazioni Ensemble-Average. Il codice deve anche prevedere la
  costruzione della funzione densità di probabilità (area normalizzata ad 1) e della funzione
  di autocorrelazione. Fate in modo che i parametri rilevanti della simulazione siano inseriti
  dall’esterno, per esempio tramite lettura da file che contenga tutti i parametri.
  
  → In questa versione, tutti i parametri sono definiti direttamente in questo script.
"""

import numpy as np
import matplotlib.pyplot as plt

def drift_OU(x, params):
    gamma = params[0]
    return -gamma * x

def drift_RISK(x, params):
    alpha, beta = params
    return -alpha * x / (1 + beta * x**2)

def euler_maruyama(x0, drift_fn, drift_params, sigma, dt, N):
    x = np.empty(N)
    x[0] = x0
    sqrt_dt = np.sqrt(dt)
    for i in range(1, N):
        dW = np.random.normal(0, sqrt_dt)
        x[i] = x[i-1] + drift_fn(x[i-1], drift_params)*dt + sigma*dW
    return x

def ensemble_average(drift_fn, drift_params, sigma, dt, N, M, x0, bins=50, xlim=(-5,5)):
    hist_all = np.zeros((M, bins))
    ac_all   = np.zeros((M, N))
    edges = np.linspace(xlim[0], xlim[1], bins+1)

    for m in range(M):
        traj = euler_maruyama(x0, drift_fn, drift_params, sigma, dt, N)
        hist_all[m], _ = np.histogram(traj, bins=edges, density=True)
        x = traj - traj.mean()
        var = np.var(traj)
        ac_all[m] = np.correlate(x, x, mode='full')[N-1:] / (var * N)

    centers = (edges[:-1] + edges[1:]) / 2
    return (centers,
            hist_all.mean(0), hist_all.std(0),
            ac_all.mean(0),   ac_all.std(0))

def plot_and_save(center, h_mean, h_std, ac_mean, ac_std, dt, title, out_prefix):
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.errorbar(center, h_mean, yerr=h_std, fmt='o')
    plt.title(f'pdf {title}')
    plt.xlabel('x'); plt.ylabel('densità')

    plt.subplot(1,2,2)
    tau = np.arange(len(ac_mean)) * dt
    # Consideriamo tutti i punti, ma per y log evita zero o negativi
    ac_plot = np.clip(ac_mean, 1e-15, None)  # evita log(0)
    ac_std_plot = ac_std

    plt.errorbar(tau, ac_plot, yerr=ac_std_plot, fmt='-')
    plt.yscale('log')
    plt.title(f'C(τ) {title} (lin-log)')
    plt.xlabel('τ')
    plt.ylabel('autocorr. (log scale)')
    plt.tight_layout()
    plt.show()

    with open(f'{out_prefix}_autocorr.dat', 'w') as f:
        for i in range(len(ac_mean)):
            f.write(f"{i}\t{ac_mean[i]:.6e}\t{ac_std[i]:.6e}\n")


def main():
    # --- Parametri definiti internamente ---
    # Ornstein–Uhlenbeck: [gamma, sigma, dt, N, M, x0]
    params_OU = [0.7, 1.0, 0.01, 1000, 100, 0.0]
    gamma, sigma_ou, dt_ou, N_ou, M_ou, x0_ou = params_OU

    center, hm, hs, acm, acs = ensemble_average(
        drift_OU, [gamma], sigma_ou, dt_ou, N_ou, M_ou, x0_ou
    )
    plot_and_save(center, hm, hs, acm, acs, dt_ou, '(OU)', 'OU')

    # Risken multiscala: [alpha, beta, sigma, dt, N, M, x0]
    params_RISK = [1.0, 0.5, 1.0, 0.01, 1000, 100, 0.0]
    alpha, beta, sigma_r, dt_r, N_r, M_r, x0_r = params_RISK

    center, hm, hs, acm, acs = ensemble_average(
        drift_RISK, [alpha, beta], sigma_r, dt_r, N_r, M_r, x0_r
    )
    plot_and_save(center, hm, hs, acm, acs, dt_r, '(Risken)', 'RISK')

if __name__ == '__main__':
    main()

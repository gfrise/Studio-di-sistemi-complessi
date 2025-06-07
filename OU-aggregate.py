"""
PROGRAMMAZIONE

Compito 1: simula Ornstein-Uhlenbeck con equazione di Langevin, calcola densità di probabilità e autocorrelazione,
parametri definiti internamente, media e deviazione standard su M=100 iterazioni.

Compito 2: simula processo multiscala con drift non lineare (Risken), stesso output e media su M iterazioni.
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_OU(theta, sigma, dt, N):
    x = np.zeros(N)
    for i in range(1, N):
        x[i] = x[i-1] - theta*x[i-1]*dt + sigma*np.random.normal(0,np.sqrt(dt))
    return x

def simulate_RISKEN(alpha, beta, sigma, dt, N):
    x = np.zeros(N)
    for i in range(1, N):
        drift = -alpha * x[i-1]/(1 + beta*x[i-1]**2)
        x[i] = x[i-1] + drift*dt + sigma*np.random.normal(0,np.sqrt(dt))
    return x

def autocorr(x):
    x = x - np.mean(x)
    var = np.var(x)
    corr = np.correlate(x, x, mode='full')[len(x)-1:] / (var*len(x))
    return corr

def analyze(sim_func, params, bins=50, xlim=(-5,5)):
    *par, N, M = params
    N, M = int(N), int(M)
    hist_all = np.zeros((M, bins))
    acorr_all = np.zeros((M, N))

    edges = np.linspace(xlim[0], xlim[1], bins+1)
    for m in range(M):
        x = sim_func(*par, N)
        hist_all[m], _ = np.histogram(x, bins=edges, density=True)
        acorr_all[m] = autocorr(x)

    centers = (edges[:-1] + edges[1:]) / 2
    return centers, hist_all.mean(0), hist_all.std(0), acorr_all.mean(0), acorr_all.std(0)

def plot_res(centers, h_mean, h_std, ac_mean, ac_std, dt, title):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.errorbar(centers, h_mean, yerr=h_std, fmt='o')
    plt.title(f'Densità di probabilità {title}')
    plt.xlabel('x')
    plt.ylabel('pdf')

    plt.subplot(1,2,2)
    tau = np.arange(200)*dt
    plt.errorbar(tau, ac_mean[:200], yerr=ac_std[:200])
    plt.title(f'Autocorrelazione {title}')
    plt.xlabel('tau')
    plt.ylabel('C(tau)')

    plt.tight_layout()
    plt.show()

def main():
    # Parametri Ornstein-Uhlenbeck: theta, sigma, dt, N passi, M iterazioni
    params_OU = [0.7, 1.0, 0.01, 1000, 100]

    centers, hm, hs, acm, acs = analyze(simulate_OU, params_OU)
    plot_res(centers, hm, hs, acm, acs, params_OU[2], '(OU)')

    # Parametri Risken: alpha, beta, sigma, dt, N passi, M iterazioni
    params_RISKEN = [1.0, 0.5, 1.0, 0.01, 1000, 100]

    centers, hm, hs, acm, acs = analyze(simulate_RISKEN, params_RISKEN)
    plot_res(centers, hm, hs, acm, acs, params_RISKEN[3], '(Risken)')

if __name__ == '__main__':
    main()

"""
sde_stability.py

Studio numerico della stabilità per l'equazione di Ornstein–Uhlenbeck:
    dx = λ x dt + σ dW(t)

Verifica di:
 1) Stabilità in media: E[X(t)] → 0 se λ < 0 e (1+λΔt) < 1
 2) Stabilità in media quadratica: E[X(t)^2] → σ^2/(-2λ)

Usare: python sde_stability.py
"""
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Parametri --------------------
lam = -0.5         # λ < 0 per stabilità
sigma = 1.0        # intensità del rumore dW
dt = 0.01          # passo temporale
T = 10.0           # tempo totale
N = int(T / dt)    # numero di passi temporali
M = 1000           # numero di realizzazioni per ensemble
x0 = 1.0           # condizione iniziale
# ---------------------------------------------------

def simulate_ou_euler(lam, sigma, dt, N, x0):
    """
    Genera una singola traiettoria OU con schema di Eulero:
    X_{n+1} = X_n + λ X_n dt + σ ΔW_n
    """
    X = np.empty(N)
    X[0] = x0
    for n in range(N-1):
        dW = np.random.normal(0, np.sqrt(dt))
        X[n+1] = X[n] + lam * X[n] * dt + sigma * dW
    return X

# Preallocazione per statistiche
mean_ensemble = np.zeros(N)
mean_sq_ensemble = np.zeros(N)

# Ensemble Monte Carlo
def run_ensemble():
    for _ in range(M):
        traj = simulate_ou_euler(lam, sigma, dt, N, x0)
        mean_ensemble[:] += traj
        mean_sq_ensemble[:] += traj**2
    mean_ensemble[:] /= M
    mean_sq_ensemble[:] /= M

# Esecuzione
run_ensemble()

time = np.linspace(0, T, N)
steady_var = sigma**2 / (-2.0 * lam)

# Plot risultati
plt.figure()
plt.plot(time, mean_ensemble, label=r'$E[X(t)]$')
plt.axhline(0, color='gray', linestyle=':')
plt.xlabel('Tempo')
plt.ylabel('Media')
plt.title('Stabilità in media (Euler)')
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(time, mean_sq_ensemble, label=r'$E[X^2(t)]$')
plt.axhline(steady_var, color='gray', linestyle=':',
            label=r'$\sigma^2/(-2\lambda)$ teorico')
plt.xlabel('Tempo')
plt.ylabel('Media quadratica')
plt.title('Stabilità in media quadratica (Euler)')
plt.grid(True)
plt.legend()
plt.show()

# Salvataggio su file
np.savetxt('mean_X.dat', np.column_stack((time, mean_ensemble)))
np.savetxt('mean_sq_X.dat', np.column_stack((time, mean_sq_ensemble)))

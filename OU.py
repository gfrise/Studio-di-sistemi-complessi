import numpy as np
import matplotlib.pyplot as plt

# --- Parametri ---
gamma1 = 0.1
dt     = 0.01
NN     = 100_000
step   = 10
# Generico tau_max “in tempo discreto”; verrà limitato dentro la funzione
tau_max = 200  

# --- Simulazione OU (Euler–Maruyama) ---
X = np.empty(NN+1)
X[0] = 0.1
eta = np.random.normal(0, np.sqrt(2.0), size=NN)

for i in range(1, NN+1):
    X[i] = X[i-1] - gamma1*X[i-1]*dt + np.sqrt(dt)*eta[i-1]

# Sotto-campioniamo
uno = X[::step]

# Statistiche
print("Media empirica:",  np.mean(uno))
print("Varianza empirica:", np.var(uno))
print("Tempo di decadimento teorico 1/gamma1:", 1/gamma1)

# --- Funzione autocorrelazione robusta ---
def autocorr(x, max_lag):
    N = len(x)
    x = x - x.mean()
    var = x.var()
    L = min(max_lag, N-1)
    result = np.empty(L+1)
    for tau in range(L+1):
        result[tau] = np.dot(x[:N-tau], x[tau:]) / ((N-tau) * var)
    return result

# Calcolo ACF empirica
acf_emp = autocorr(uno, tau_max)

# Assi dei ritardi in unità di tempo reale
taus = np.arange(len(acf_emp)) * dt * step

# --- Plot comparativo ---
plt.figure(figsize=(8,5))
plt.semilogy(taus, np.exp(-gamma1*taus),   label=r"$e^{-\gamma_1\tau}$", color="red")
plt.semilogy(taus, acf_emp,                label="ACF empirica",
             marker="o", linestyle="none", markersize=4)
plt.xlabel(r"$\tau$")
plt.ylabel(r"$\rho(\tau)$")
plt.title("ACF teoretica vs empirica del processo OU")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()

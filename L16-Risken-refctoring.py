import numpy as np
import matplotlib.pyplot as plt

# ===================== Parametri =====================
T = 1000          # Lunghezza della serie sottocampionata
kappa = 1.0       # Drift
x0 = 0.0          # Condizione iniziale
seed = 42         # Seme RNG
enne = 100        # Passi per unità di tempo
tmax = 100        # Lag massimo autocorrelazione
nbin = 50         # Numero bin PDF
# ====================================================

np.random.seed(seed)
dt = 1.0 / enne
N = T * enne  # Lunghezza totale

# Genera rumore gaussiano
Z = np.random.normal(loc=0.0, scale=np.sqrt(2 * dt), size=N)

# Simulazione con Eulero-Maruyama
x = np.zeros(N + 1)
x[0] = x0
for i in range(N):
    drift = kappa * dt if x[i] < 0 else -kappa * dt
    x[i + 1] = x[i] + drift + Z[i]

# Statistiche su tutta la traiettoria
mean_x = np.mean(x)
var_x = np.var(x)
print(f"Media totale x(t): {mean_x:.6f}")
print(f"Varianza totale x(t): {var_x:.6f}")

# Salvataggio dati completi
np.savetxt("numeri_RISKEN.dat", np.column_stack((np.arange(N+1), x)))

# Sottocampionamento: un punto ogni unità di tempo
y = x[::enne][:T]

mean_y = np.mean(y)
var_y = np.var(y)
print(f"Media y(t): {mean_y:.6f}")
print(f"Varianza y(t): {var_y:.6f}")

# PDF normalizzata
i_vals, edges = np.histogram(y, bins=nbin, density=True)
centers = 0.5 * (edges[:-1] + edges[1:])
np.savetxt("pdf_RISKEN.dat", np.column_stack((centers, i_vals)))

plt.figure()
plt.plot(centers, i_vals, drawstyle='steps-mid')
plt.xlabel("x")
plt.ylabel("PDF")
plt.title("Densità di Probabilità")
plt.grid(True)
plt.tight_layout()
plt.show()

# Autocorrelazione manuale
acf = np.zeros(tmax)
for lag in range(tmax):
    cov = np.mean((y[:T-lag] - mean_y) * (y[lag:] - mean_y))
    acf[lag] = cov / var_y

np.savetxt("acf_RISKEN.dat", np.column_stack((np.arange(tmax), acf)))

plt.figure()
plt.plot(np.arange(tmax), acf)
plt.xlabel("Lag")
plt.ylabel("Autocorrelazione")
plt.title("Funzione di Autocorrelazione")
plt.grid(True)
plt.tight_layout()
plt.show()

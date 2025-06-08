import numpy as np
import matplotlib.pyplot as plt

gamma = 1.0     # coefficiente di ritorno verso la media
enne = 100      # punti per unità di tempo
x0 = 0.0        # valore iniziale
nR = 1000       # numero di punti nella serie sottocampionata
nbin = 50       # numero di bin per l'istogramma
tmax = 100      # lag massimo per l'autocorrelazione
dt = 1.0 / enne
N = int(nR * enne)  # numero di punti totali simulati

# === SIMULAZIONE DEL PROCESSO OU ===
t = np.arange(N) * dt
X = np.empty(N)
X[0] = x0

for i in range(1, N):
    noise = np.random.randn()*np.sqrt(2*gamma*dt)
    X[i] = X[i-1]-gamma*X[i-1]*dt+noise

# === SOTTOCAMPIONAMENTO (un punto ogni unità di tempo) ===
y = X[::enne]

# === STATISTICHE DI BASE ===
d_media = y.mean()
d_var = y.var()
print(f"Media: {d_media:.6f}, Varianza: {d_var:.6f}")

# === DENSITÀ DI PROBABILITÀ (PDF) ===
i_vals, edges = np.histogram(y, bins=nbin, density=True)
centers = 0.5 * (edges[:-1] + edges[1:])

plt.figure()
plt.plot(centers, i_vals)
plt.xlabel('X')
plt.ylabel('PDF')
plt.title('Densità di Probabilità')
plt.grid(True)
plt.tight_layout()
plt.show()

# === AUTOCORRELAZIONE MANUALE ===
T = len(y)
acf = np.zeros(tmax)

for lag in range(tmax):
    cov = np.sum((y[:T - lag] - d_media) * (y[lag:] - d_media))
    acf[lag] = cov / ((T - lag) * d_var)

plt.figure()
plt.plot(np.arange(tmax), acf)
plt.xlabel('Lag')
plt.ylabel('Autocorrelazione')
plt.title('Funzione di Autocorrelazione')
plt.grid(True)
plt.tight_layout()
plt.show()

"""
sde_integrators.py

Implementazione di due schemi di integrazione per equazioni differenziali stocastiche:
 1) Metodo di Eulero (ordine 1)
 2) Metodo di Eulero avanzato (ordine 2, approssimazione di Taylor)

SDE generica: dx = h(x)*dt + g(x)*dW

Usare: python sde_integrators.py

Metodo di Eulero avanzato:
---------------------------
Lo schema di Eulero avanzato si basa su una espansione di Taylor stocastica dell'equazione di Langevin.
Aggiunge un termine correttivo che approssima l'evoluzione del drift h(x) su un intervallo dt, includendo l'effetto
della curvatura (derivata prima del drift).

Formula:
    x_{n+1} = x_n + h(x_n) * dt + g(x_n) * ΔW + 0.5 * h'(x_n) * g(x_n) * (ΔW^2 - 2dt)

Dove:
 - ΔW è un incremento di Wiener: ΔW ~ N(0, 2dt)
 - h'(x) è la derivata del drift h rispetto a x

Il termine extra migliora la precisione, specialmente quando h(x) varia rapidamente.
"""
"""
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Definizione SDE --------------------
def h(x):
    """Deriva (drift)"""
    return -x

def g(x):
    """Diffusione (diffusion coefficient)"""
    return 1.0

def h_prime(x):
    """Derivata di h rispetto a x"""
    return -1.0

def g_prime(x):
    """Derivata di g rispetto a x"""
    return 0.0

# -------------------- Parametri simulazione --------------------
dt = 0.01     # passo temporale
t_final = 10.0
N = int(t_final / dt)
x0 = 1.0      # condizione iniziale
seed = 12345  # seme per RNG
np.random.seed(seed)

# -------------------- Funzione integratore --------------------
def simulate_euler(x0, dt, N):
    x = np.zeros(N)
    x[0] = x0
    for i in range(N-1):
        dw = np.sqrt(2 * dt) * np.random.randn()
        x[i+1] = x[i] + h(x[i]) * dt + g(x[i]) * dw
    return x


def simulate_advanced_euler(x0, dt, N):
    x = np.zeros(N)
    x[0] = x0
    for i in range(N-1):
        xi = x[i]
        dw = np.sqrt(2 * dt) * np.random.randn()
        # Termini fino a secondo ordine (Taylor)
        x[i+1] = (xi
                  + h(xi) * dt
                  + g(xi) * dw
                  + 0.5 * h_prime(xi) * g(xi) * (dw**2 - 2 * dt)
                 )
    return x

# -------------------- Esecuzione e plot --------------------
time = np.linspace(0, t_final, N)

x_euler = simulate_euler(x0, dt, N)
x_adv = simulate_advanced_euler(x0, dt, N)

plt.figure()
plt.plot(time, x_euler, label='Euler ordine 1', alpha=0.7)
plt.plot(time, x_adv, label='Euler avanzato ordine 2', alpha=0.7)
plt.xlabel('Tempo')
plt.ylabel('x(t)')
plt.title('Confronto metodi di integrazione SDE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Salva risultati su file
# np.savetxt('traj_euler.dat', np.column_stack((time, x_euler)))
# np.savetxt('traj_adv_euler.dat', np.column_stack((time, x_adv)))

"""Fine implementazione"""

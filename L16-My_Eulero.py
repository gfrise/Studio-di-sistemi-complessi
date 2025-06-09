import numpy as np
import matplotlib.pyplot as plt

# Definizione dei termini della SDE
h = lambda x: -35 * x
h_prime = lambda x: -35

g = lambda x: 1.0

dt, tf, x0 = 0.1, 10.0, 1.0
N = int(tf / dt)
time = np.linspace(0, tf, N)

np.random.seed(12345)
dW = np.sqrt(2 * dt) * np.random.randn(N - 1)

def integrate(x0, dt, dW, advanced=False):
    x = np.empty(len(dW) + 1)
    x[0] = x0
    for i, dw in enumerate(dW):
        xi = x[i]
        dx = h(xi) * dt + g(xi) * dw
        if advanced:
            dx += 0.5 * h_prime(xi) * g(xi) * (dw**2 - 2 * dt)
        x[i + 1] = xi + dx
    return x

# Simulazioni
x_euler = integrate(x0, dt, dW, advanced=False)
x_adv   = integrate(x0, dt, dW, advanced=True)

# Grafico dei risultati
plt.figure(figsize=(10, 5))
plt.plot(time, x_euler, label='Euler (1° ordine)', alpha=0.7)
plt.plot(time, x_adv, label='Euler avanzato (2° ordine)', alpha=0.7)
plt.xlabel("Tempo")
plt.ylabel("x(t)")
plt.title("Confronto tra integratori SDE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_sequence(filename):
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    with open(filename) as f:
        return [base_to_idx[line.strip()] for line in f if line.strip() in base_to_idx]

@njit
def log_sum_exp(log_arr):
    max_val = np.max(log_arr)
    if max_val == -np.inf:
        return -np.inf
    s = 0.0
    for val in log_arr:
        s += np.exp(val - max_val)
    return max_val + np.log(s)

@njit(inline='always')
def log_add_exp(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a > b:
        return a + np.log1p(np.exp(b - a))
    else:
        return b + np.log1p(np.exp(a - b))

@njit
def forward_log(O, pi_log, A_log, B_log):
    T = len(O) - 1
    N = A_log.shape[0]
    
    alpha = np.full((T, N), -np.inf)

    # inizializzazione
    for i in range(N):
        alpha[0, i] = pi_log[i] + B_log[i, O[0], O[1]]

    # ricorsione
    for t in range(1, T):
        for j in range(N):
            temp = np.empty(N)
            for i in range(N):
                temp[i] = alpha[t-1, i] + A_log[i, j]
            alpha[t, j] = log_sum_exp(temp) + B_log[j, O[t], O[t+1]]

    return alpha

@njit
def backward_log(O, A_log, B_log):
    T = len(O) - 1
    N = A_log.shape[0]
    beta = np.full((T, N), -np.inf)

    # inizializzazione
    for i in range(N):
        beta[T-1, i] = 0.0  # log(1)

    # ricorsione backward
    for t in range(T-2, -1, -1):
        for i in range(N):
            temp = np.empty(N)
            for j in range(N):
                temp[j] = A_log[i, j] + B_log[j, O[t+1], O[t+2]] + beta[t+1, j]
            beta[t, i] = log_sum_exp(temp)

    return beta

@njit
def baum_welch_step_log(O, pi_log, A_log, B_log):
    T = len(O) - 2
    alpha = forward_log(O, pi_log, A_log, B_log)
    beta = backward_log(O, A_log, B_log)
    N = pi_log.shape[0]

    gamma = np.empty((T, N))
    for t in range(T):
        denom = log_sum_exp(alpha[t] + beta[t])
        for i in range(N):
            gamma[t, i] = alpha[t, i] + beta[t, i] - denom

    xi = np.full((T-1, N, N), -np.inf)
    for t in range(T-1):
        denom = -np.inf
        for i in range(N):
            for j in range(N):
                val = alpha[t, i] + A_log[i, j] + B_log[j, O[t+1], O[t+2]] + beta[t+1, j]
                xi[t, i, j] = val
                denom = log_add_exp(denom, val)
        for i in range(N):
            for j in range(N):
                xi[t, i, j] -= denom

    # aggiorna pi e normalizza
    pi_log_new = gamma[0]
    pi_log_new -= log_sum_exp(pi_log_new)

    # aggiorna A e normalizza
    A_log_new = np.empty_like(A_log)
    for i in range(N):
        numer_arr = np.full(N, -np.inf)
        denom = -np.inf
        for j in range(N):
            numer = -np.inf
            for t in range(T-1):
                numer = log_add_exp(numer, xi[t, i, j])
            numer_arr[j] = numer
        for t in range(T-1):
            denom = log_add_exp(denom, gamma[t, i])
        A_log_new[i] = numer_arr - denom
        norm = log_sum_exp(A_log_new[i])
        A_log_new[i] -= norm

    # aggiorna B e normalizza riga per riga
    B_log_new = np.empty_like(B_log)
    for i in range(N):
        numer_mat = np.full((4, 4), -np.inf)
        denom_per_row = np.full(4, -np.inf)
    
        for p in range(4):
            for q in range(4):
                numer = -np.inf
                for t in range(T):
                    if O[t] == p and O[t+1] == q:
                        numer = log_add_exp(numer, gamma[t, i])
                numer_mat[p, q] = numer
                denom_per_row[p] = log_add_exp(denom_per_row[p], numer)
    
        B_log_new[i] = numer_mat
    
        # normalizza riga per riga
        for p in range(4):
            B_log_new[i, p] -= log_sum_exp(B_log_new[i, p])

    return pi_log_new, A_log_new, B_log_new, alpha

N = 2

np.random.seed(42)

stay = np.random.uniform(0.65,0.95)
A = np.array([[stay, 1-stay],
              [1-stay, stay]])
A_log = np.log(A)

B = np.zeros((2, 4, 4))

for s in range(2):
    for prev in range(4):
        probs = np.ones(4)

        if s == 0:
            # Penalizza C/G → moltiplica A/T (0,3) per 1.5
            probs[0] *= np.random.uniform(1.0,10.0)  # A
            probs[3] *= np.random.uniform(1.0,10.0)  # T
        else:
            # Favorisce C/G → moltiplica C/G (1,2) per 1.5
            probs[1] *= np.random.uniform(1.0,10.0)  # C
            probs[2] *= np.random.uniform(1.0,10.0)  # G

        B[s, prev] = probs / probs.sum()

B_log = np.log(B)

pi = np.array([0.5, 0.5])
pi_log = np.log(pi)

observations = load_sequence("observations.txt")

log_likelihoods = []
max_iter = 500
epsilon = 1e-4
prev_ll = -np.inf

for _ in tqdm(range(max_iter)):
    pi_log, A_log, B_log, alpha = baum_welch_step_log(observations, pi_log, A_log, B_log)
    
    # Calcola log-likelihood da alpha
    ll = log_sum_exp(alpha[-1])
    log_likelihoods.append(ll)

    if np.abs(ll - prev_ll) < epsilon:
        print(f"Convergenza raggiunta: Δll = {ll - prev_ll:.5f}")
        break

    prev_ll = ll

pi_est = np.exp(pi_log)
A_est = np.exp(A_log)
B_est = np.exp(B_log)

print("Stima dei vettori di probabilità iniziali (pi):")
print(pi_est)

print("\nStima matrice di transizione (A):")
print(A_est)

print("\nStima matrice di emissione (B):")
print(B_est)

plt.figure(figsize=(10, 5))
plt.plot(log_likelihoods, marker='.')
plt.title("Andamento del Log-Likelihood durante Baum-Welch")
plt.xlabel("Iterazione")
plt.ylabel("Log-Likelihood")
plt.grid(True)
plt.tight_layout()
plt.savefig("Grafici/Baum-Welch.png")
plt.show()

np.save("pi_est.npy", pi_est)
np.save("A_est.npy", A_est)
np.save("B_est.npy", B_est)
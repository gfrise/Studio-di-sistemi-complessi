import numpy as np
from numba import njit
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
def load_sequence(filename):
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    with open(filename) as f:
        return [base_to_idx[line.strip()] for line in f if line.strip() in base_to_idx]
@njit
def log_sum_exp(log_probs):
    max_log = np.max(log_probs)
    return max_log + np.log(np.sum(np.exp(log_probs - max_log)))
@njit
def forward_log(O, pi_log, A_log, B_log):
    N = pi_log.shape[0]
    T = len(O) - 1
    alpha = np.full((T, N), -np.inf)

    for i in range(N):
        if O[0] == -1 or O[1] == -1:
            emission = np.max(B_log[i])
        else:
            emission = B_log[i, O[0], O[1]]
        alpha[0, i] = pi_log[i] + emission

    for t in range(1, T):
        for j in range(N):
            temp = alpha[t-1] + A_log[:, j]
            if O[t] == -1 or O[t+1] == -1:
                emission = np.max(B_log[j])
            else:
                emission = B_log[j, O[t], O[t+1]]
            alpha[t, j] = log_sum_exp(temp) + emission

    return alpha
@njit
def backward_log(O, A_log, B_log):
    N = A_log.shape[0]
    T = len(O) - 1
    beta = np.full((T, N), -np.inf)
    beta[-1, :] = 0.0

    for t in range(T - 2, -1, -1):
        for i in range(N):
            temp = np.empty(N)
            for j in range(N):
                if O[t+1] == -1 or O[t+2] == -1:
                    emission = np.max(B_log[j])
                else:
                    emission = B_log[j, O[t+1], O[t+2]]
                temp[j] = A_log[i, j] + emission + beta[t+1, j]
            beta[t, i] = log_sum_exp(temp)

    return beta
@njit
def posterior_decoding_log(O, pi_log, A_log, B_log):
    alpha = forward_log(O, pi_log, A_log, B_log)
    beta = backward_log(O, A_log, B_log)
    T, N = alpha.shape
    posterior = np.zeros((T, N))

    for t in range(T):
        log_gamma = alpha[t] + beta[t]
        log_gamma -= log_sum_exp(log_gamma)  # normalizzazione log
        posterior[t] = np.exp(log_gamma)

    states = np.argmax(posterior, axis=1)
    return states
observations = load_sequence("observations.txt")
true_states = np.loadtxt("states.txt", dtype=int)

pi_log = np.log(np.load("pi_est.npy"))
A_log = np.log(np.load("A_est.npy"))
B_log = np.log(np.load("B_est.npy"))

# Usa posterior decoding invece di Viterbi
predicted_states = posterior_decoding_log(observations, pi_log, A_log, B_log)

# Calcola accuratezza sugli stati
accuracy = np.mean(predicted_states == true_states[:-1])
print(f"Accuratezza Posterior: {accuracy:.4f}")
state_names = ["Non-CpG", "CpG"]

conf_mat = confusion_matrix(true_states[:-1], predicted_states)

# Accuratezza per ogni stato reale
state_counts = conf_mat.sum(axis=1)  # Totale occorrenze per ogni stato reale
correct_preds = np.diag(conf_mat)    # Predizioni corrette (diagonale)
per_state_accuracy = correct_preds / state_counts

print("Accuratezza per stato:")
for i, acc in enumerate(per_state_accuracy):
    print(f"  {state_names[i]}: {acc:.4f} ({correct_preds[i]}/{state_counts[i]})")

plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"Pred {s}" for s in state_names],
            yticklabels=[f"True {s}" for s in state_names])
plt.title("Matrice di Confusione - Posterior Decoding")
plt.xlabel("Stati Predetti")
plt.ylabel("Stati Veri")
plt.tight_layout()
plt.savefig("Grafici/Posterior-state.png")
plt.show()
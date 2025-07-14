import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from sklearn.metrics import confusion_matrix
def load_sequence(filename):
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    with open(filename) as f:
        return [base_to_idx[line.strip()] for line in f if line.strip() in base_to_idx]
def introduce_gaps(seq, gap_ratio):
    seq_with_gaps = np.array(seq)  # conversione sicura in array
    n = len(seq_with_gaps)
    n_gaps = int(n * gap_ratio)
    gap_positions = np.random.choice(n, n_gaps, replace=False)
    seq_with_gaps[gap_positions] = -1
    return seq_with_gaps, gap_positions
@njit
def viterbi_impute_log(O, pi_log, A_log, B_log, bases=4):
    N = pi_log.shape[0]
    T = len(O) - 1

    delta = np.full((T, N), -1e9)  # log-space inizializzato a -inf
    psi = np.zeros((T, N), dtype=np.int32)

    # Inizializzazione
    for i in range(N):
        if O[0] == -1 or O[1] == -1:
            emission_prob = np.max(B_log[i])
        else:
            emission_prob = B_log[i, O[0], O[1]]
        delta[0, i] = pi_log[i] + emission_prob

    # Ricorsione
    for t in range(1, T):
        for j in range(N):
            max_val = -1e9
            argmax_k = 0
            for k in range(N):
                val = delta[t-1, k] + A_log[k, j]
                if val > max_val:
                    max_val = val
                    argmax_k = k
            psi[t, j] = argmax_k

            if O[t] == -1 or O[t+1] == -1:
                emission_prob = np.max(B_log[j])
            else:
                emission_prob = B_log[j, O[t], O[t+1]]

            delta[t, j] = max_val + emission_prob

    # Backtracking
    states = np.zeros(T, dtype=np.int32)
    best_last = 0
    max_final = -1e9
    for i in range(N):
        if delta[T-1, i] > max_final:
            max_final = delta[T-1, i]
            best_last = i
    states[T-1] = best_last

    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]

    # Imputazione
    imputed_obs = np.copy(O)

    for t in range(len(O)):
        if imputed_obs[t] == -1:
            if t == 0:
                probs = np.empty(bases)
                for i in range(bases):
                    max_val = -1e9
                    for j in range(bases):
                        val = B_log[states[0], i, j]
                        if val > max_val:
                            max_val = val
                    probs[i] = max_val
                best_base = 0
                max_prob = -1e9
                for b in range(bases):
                    if probs[b] > max_prob:
                        max_prob = probs[b]
                        best_base = b
                imputed_obs[t] = best_base
            else:
                s = states[t-1]
                prev_base = imputed_obs[t-1]
                probs = B_log[s, prev_base]
                best_base = 0
                max_prob = -1e9
                for b in range(bases):
                    if probs[b] > max_prob:
                        max_prob = probs[b]
                        best_base = b
                imputed_obs[t] = best_base

    return imputed_obs, states
observations = load_sequence("observations.txt")
true_seq = np.array(observations)  # attenzione qui: è un np.array ora

pi_log = np.log(np.load("pi_est.npy"))
A_log = np.log(np.load("A_est.npy"))
B_log = np.log(np.load("B_est.npy"))

gap_ratios = np.array((0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0))
mean_accuracies = []
std_accuracies = []

n_iter = 50  # numero di iterazioni per ogni gap_ratio

for gap_ratio in tqdm(gap_ratios, desc="Gap levels"):
    accs = []
    for _ in range(n_iter):
        seq_with_gaps, gap_positions = introduce_gaps(true_seq, gap_ratio)

        imputed_obs, states = viterbi_impute_log(seq_with_gaps, pi_log, A_log, B_log)

        true_bases = true_seq[gap_positions]
        imputed_bases = np.array(imputed_obs)[gap_positions]

        acc = np.mean(true_bases == imputed_bases)
        accs.append(acc)

    mean_accuracies.append(np.mean(accs))
    std_accuracies.append(np.std(accs))

    print(f"Gap ratio: {gap_ratio*100:.3f}% - Media: {np.mean(accs):.3f}, Std: {np.std(accs):.2e}")
plt.figure(figsize=(10, 6))
plt.semilogx()
plt.errorbar(gap_ratios * 100, mean_accuracies, yerr=std_accuracies, linestyle='--', fmt='o', capsize=5)
plt.xlabel("Percentuale di gap (%)")
plt.ylabel("Accuratezza media")
plt.title("Accuratezza vs % di gap (con Viterbi imputazione)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Grafici/Viterbi-base1.png")
plt.show()
gap_ratio = 0.01

seq_with_gaps, gap_positions = introduce_gaps(true_seq, gap_ratio)
imputed_obs, states = viterbi_impute_log(seq_with_gaps, pi_log, A_log, B_log)

# Estrai basi vere e imputate solo nei gap
true_bases = true_seq[gap_positions]
imputed_bases = np.array(imputed_obs)[gap_positions]

# Confusion matrix
labels = [0, 1, 2, 3]  # A, C, G, T
base_names = ['A', 'C', 'G', 'T']
cm = confusion_matrix(true_bases, imputed_bases, labels=labels)

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=base_names, yticklabels=base_names)
plt.xlabel("Basi imputate")
plt.ylabel("Basi reali")
plt.title(f"Confusion Matrix Viterbi - Gap {int(gap_ratio*100)}%")
plt.savefig("Grafici/Viterbi-base2.png")
plt.show()


print("\nAccuratezza per base:")
for i, base in enumerate(base_names):
    total = cm[i].sum()
    correct = cm[i, i]
    acc = correct / total if total > 0 else 0.0
    print(f"{base}: {acc:.4f}  ({correct}/{total})")
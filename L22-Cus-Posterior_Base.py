import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
def load_sequence(filename):
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    with open(filename) as f:
        return [base_to_idx[line.strip()] for line in f if line.strip() in base_to_idx]
@njit
def log_sum_exp(log_probs):
    max_log = np.max(log_probs)
    s = 0.0
    for lp in log_probs:
        s += np.exp(lp - max_log)
    return max_log + np.log(s)
@njit
def forward_log(O, pi_log, A_log, B_log):
    N = pi_log.shape[0]
    bases = B_log.shape[1]
    T = len(O) - 1

    alpha = np.full((T, N), -np.inf)

    # init
    for i in range(N):
        if O[0] == -1 or O[1] == -1:
            max_emit = -1e10
            for b1 in range(bases):
                for b2 in range(bases):
                    if B_log[i, b1, b2] > max_emit:
                        max_emit = B_log[i, b1, b2]
            emission = max_emit
        else:
            emission = B_log[i, O[0], O[1]]
        alpha[0, i] = pi_log[i] + emission

    # recursion
    for t in range(1, T):
        for j in range(N):
            max_val = -1e10
            for i_ in range(N):
                val = alpha[t-1, i_] + A_log[i_, j]
                if val > max_val:
                    max_val = val
            if O[t] == -1 or O[t+1] == -1:
                max_emit = -1e10
                for b1 in range(bases):
                    for b2 in range(bases):
                        if B_log[j, b1, b2] > max_emit:
                            max_emit = B_log[j, b1, b2]
                emission = max_emit
            else:
                emission = B_log[j, O[t], O[t+1]]
            alpha[t, j] = max_val + emission

    return alpha
@njit
def backward_log(O, A_log, B_log):
    N = A_log.shape[0]
    bases = B_log.shape[1]
    T = len(O) - 1

    beta = np.full((T, N), -np.inf)

    # init
    for i in range(N):
        beta[T-1, i] = 0.0

    # recursion
    for t in range(T-2, -1, -1):
        for i_ in range(N):
            max_val = -1e10
            for j in range(N):
                emission = 0.0
                if O[t+1] == -1 or O[t+2] == -1:
                    max_emit = -1e10
                    for b1 in range(bases):
                        for b2 in range(bases):
                            if B_log[j, b1, b2] > max_emit:
                                max_emit = B_log[j, b1, b2]
                    emission = max_emit
                else:
                    emission = B_log[j, O[t+1], O[t+2]]
                val = A_log[i_, j] + emission + beta[t+1, j]
                if val > max_val:
                    max_val = val
            beta[t, i_] = max_val

    return beta
@njit
def posterior_log(alpha, beta):
    T, N = alpha.shape
    posterior = np.empty((T, N))
    for t in range(T):
        denom = log_sum_exp(alpha[t] + beta[t])
        for i in range(N):
            posterior[t, i] = alpha[t, i] + beta[t, i] - denom
    return posterior
@njit
def posterior_impute_bases(O, pi_log, A_log, B_log):
    N = pi_log.shape[0]
    bases = B_log.shape[1]
    T = len(O) - 1

    alpha = forward_log(O, pi_log, A_log, B_log)
    beta = backward_log(O, A_log, B_log)

    posterior = posterior_log(alpha, beta)

    imputed_obs = O.copy()

    for t in range(len(O)):
        if O[t] == -1:
            candidates = np.zeros(bases)
            prev_base = -1
            next_base = -1

            if t > 0:
                prev_base = imputed_obs[t-1]
            if t < len(O) - 1:
                next_base = imputed_obs[t+1]

            for b in range(bases):
                val = 0.0

                if prev_base != -1 and t-1 < T:
                    max_emit = -1e10
                    for s in range(N):
                        tmp = posterior[t-1, s] + B_log[s, prev_base, b]
                        if tmp > max_emit:
                            max_emit = tmp
                    val += max_emit
                if next_base != -1 and t < T:
                    max_emit = -1e10
                    for s in range(N):
                        tmp = posterior[t, s] + B_log[s, b, next_base]
                        if tmp > max_emit:
                            max_emit = tmp
                    val += max_emit

                candidates[b] = val

            imputed_obs[t] = np.argmax(candidates)

    return imputed_obs
def introduce_gaps(seq, gap_ratio):
    seq_with_gaps = np.array(seq)
    n = len(seq_with_gaps)
    n_gaps = int(n * gap_ratio)
    gap_positions = np.random.choice(n, n_gaps, replace=False)
    seq_with_gaps[gap_positions] = -1
    return seq_with_gaps, gap_positions
observations = load_sequence("observations.txt")
true_seq = np.array(observations)

pi_log = np.log(np.load("pi_est.npy"))
A_log = np.log(np.load("A_est.npy"))
B_log = np.log(np.load("B_est.npy"))

gap_ratios = np.array((0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0))
mean_accuracies = []
std_accuracies = []

n_iter = 50

for gap_ratio in tqdm(gap_ratios, desc="Gap levels"):
    accs = []
    for _ in range(n_iter):
        seq_with_gaps, gap_positions = introduce_gaps(true_seq, gap_ratio)
        imputed_obs = posterior_impute_bases(seq_with_gaps, pi_log, A_log, B_log)

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
plt.savefig("Grafici/Posterior-base1.png")
plt.show()
gap_ratio = 0.01
seq_with_gaps, gap_positions = introduce_gaps(true_seq, gap_ratio)
imputed_obs = posterior_impute_bases(seq_with_gaps, pi_log, A_log, B_log)

true_bases = true_seq[gap_positions]
imputed_bases = np.array(imputed_obs)[gap_positions]

labels = [0, 1, 2, 3]
base_names = ['A', 'C', 'G', 'T']
cm = confusion_matrix(true_bases, imputed_bases, labels=labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=base_names, yticklabels=base_names)
plt.xlabel("Basi imputate")
plt.ylabel("Basi reali")
plt.title(f"Confusion Matrix Posterior - Gap {int(gap_ratio*100)}%")
plt.savefig("Grafici/Posterior-base2.png")
plt.show()

print("\nAccuratezza per base:")
for i, base in enumerate(base_names):
    total = cm[i].sum()
    correct = cm[i, i]
    acc = correct / total if total > 0 else 0.0
    print(f"{base}: {acc:.4f}  ({correct}/{total})")
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
def load_sequence(filename):
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    with open(filename) as f:
        return [base_to_idx[line.strip()] for line in f if line.strip() in base_to_idx]
def viterbi_log(O, pi_log, A_log, B_log):
    T = len(O) -1
    N = pi_log.shape[0]
    
    delta = np.full((T, N), -np.inf)  # probabilità massima log di arrivare in stato i al tempo t
    psi = np.zeros((T, N), dtype=int)  # backpointer per ricostruire percorso
    
    # Initialization
    delta[0] = pi_log + B_log[:, O[0], O[1]]
    
    # Recursion
    for t in range(1, T):
        for j in range(N):
            temp_vals = delta[t-1] + A_log[:, j]
            psi[t, j] = np.argmax(temp_vals)
            delta[t, j] = np.max(temp_vals) + B_log[j, O[t], O[t+1]]
    
    # Termination
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    
    # Path backtracking
    for t in reversed(range(T-1)):
        states[t] = psi[t+1, states[t+1]]
    
    return states
observations = load_sequence("observations.txt")
true_states = np.loadtxt("states.txt", dtype=int)

pi_log = np.log(np.load("pi_est.npy"))
A_log = np.log(np.load("A_est.npy"))
B_log = np.log(np.load("B_est.npy"))

predicted_states = viterbi_log(observations, pi_log, A_log, B_log)

accuracy = np.mean(predicted_states == true_states[:-1])
print(f"Accuratezza Viterbi: {accuracy:.4f}")
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
plt.title("Matrice di Confusione - Viterbi Algorithm")
plt.xlabel("Stati Predetti")
plt.ylabel("Stati Veri")
plt.tight_layout()
plt.savefig("Grafici/Viterbi-state.png")
plt.show()
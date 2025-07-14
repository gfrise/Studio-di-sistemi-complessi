import numpy as np
np.random.seed(17)

# Parametri
n_steps = 1000000
states_labels = ['GC_poor', 'GC_rich']
bases = ['A', 'C', 'G', 'T']
base_to_idx = {b: i for i, b in enumerate(bases)}
idx_to_base = {i: b for i, b in enumerate(bases)}

# Inizializzazione π casuale
pi = np.random.dirichlet(np.ones(2))

# Matrice di transizione tra stati
A = np.array([[0.9, 0.1],
              [0.1, 0.9]])

# Matrice di emissione B[s, prev_base, next_base]
B = np.zeros((2, 4, 4))

# Stato 0 = GC-poor: penalizza C/G
# Stato 1 = GC-rich: favorisce C/G

for s in range(2):
    for prev in range(4):
        # probs = np.ones(4) + 0.3 * np.random.randn(4) # rumore gaussiano di devstd 0.3
        # probs = np.clip(probs, 0.01, None)  # minima probabilità 0.01 per evitare negativi o zero
        probs = np.random.rand(4)
        
        if s == 0:
            # Penalizza C/G → moltiplica A/T (0,3) per 4.0
            probs[0] *= 4.0  # A
            probs[3] *= 4.0  # T
        else:
            # Favorisce C/G → moltiplica C/G (1,2) per 4.0
            probs[1] *= 4.0  # C
            probs[2] *= 4.0  # G

        B[s, prev] = probs / probs.sum()

# Inizializzazione
states = []
observations = []

# Scegli il primo stato e la prima base a caso
s = np.random.choice(2, p=pi)
first_base = np.random.choice(4)  # indice tra 0 e 3
states.append(s)
observations.append(first_base)

# Generazione sequenza
for t in range(1, n_steps):
    s = np.random.choice(2, p=A[states[-1]])
    prev_base = observations[-1]
    next_base = np.random.choice(4, p=B[s, prev_base])
    states.append(s)
    observations.append(next_base)

# Salvataggio su file
with open("observations.txt", "w") as f:
    for i in observations:
        f.write(idx_to_base[i] + "\n")

with open("states.txt", "w") as f:
    for i in states:
        f.write(str(i) + "\n")
print("Vettori di probabilità iniziali (pi):")
print(pi)

print("\nMatrice di transizione (A):")
print(A)

print("\nTensore di emissione (B):")
print(B)
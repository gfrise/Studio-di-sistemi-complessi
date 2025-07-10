import numpy as np

# Parametri (equivalente a NN e iter in Mathematica)
sample_size = 100_000
n_iterations = 1000

# Dati originali (EMP in Mathematica)
original_data = np.random.normal(3.0, 0.4, sample_size)

# Inizializzazione strutture (come index[st] e datab[st] in Mathematica)
bootstrap_means = np.zeros(n_iterations)

for st in range(n_iterations):  # Equivalente al Do[... {st, 1, iter}]
    # 1. Genera indici casuali (RandomInteger[{1, NN}])
    indices = np.random.randint(0, sample_size, sample_size)  # 0-based
    #array di sample size int casuali da 0 a sample_size-1  
    
    # 2. Costruisci campione bootstrap (EMP[[index[st][[n]]]])
    bootstrap_sample = original_data[indices]
    
    # 3. Calcola media (equivalente a Mean[datab[st]])
    bootstrap_means[st] = np.mean(bootstrap_sample)

    # 4. Stampa info per i primi 10 campioni (TableForm)
    # Mostra come alcuni valori si ripetano nei campioni bootstrap (gli elementi unici saranno meno di sample_size).
    
    if st < 10:
        unique_elements = len(np.unique(bootstrap_sample))
        print(f"Campione {st+1}:")
        print(f"  Dimensione: {len(bootstrap_sample)}")
        print(f"  Elementi unici: {unique_elements}\n")

# Risultati finali
print(f"Media delle medie: {np.mean(bootstrap_means):.5f}")
print(f"Deviazione standard: {np.std(bootstrap_means):.5f}")
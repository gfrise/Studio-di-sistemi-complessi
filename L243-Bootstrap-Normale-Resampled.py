import numpy as np, matplotlib.pyplot as plt

# Parametri comuni
num_samples = 20000
sample_size = 20000
mu = 3
# CASO 3: Bootstrap da normale
sd = 0.4
sample_means3 = np.zeros(num_samples)

# Genera un solo campione originale
original_sample = np.random.normal(mu, sd, sample_size)
original_mean = np.mean(original_sample)

for i in range(num_samples):
    # Campionamento con ripetizione dall'originale
    bootstrap_sample = np.random.choice(original_sample, sample_size, replace=True)
    sample_means3[i] = np.mean(bootstrap_sample)

mean_of_means3 = np.mean(sample_means3)
std_of_means3 = np.std(sample_means3)
theoretical_std3 = np.std(original_sample) / np.sqrt(sample_size)

print("\nCASO 3: Bootstrap da Normale")
print("Media popolazione:", mu)
print("Media campione originale:", original_mean)
print("Media stimata bootstrap:", mean_of_means3)
print("Sigma teorica (s/sqrt(N)):", theoretical_std3)
print("Deviazione standard delle medie bootstrap:", std_of_means3)
plt.hist(sample_means3, bins=30)
plt.title("Caso 3: Bootstrap medie (Normale)")
plt.show()

#rispetto al caso 1
# Generazione dati: Tanti campioni indipendenti vs Un campione + ricampionamento
# Memoria: Caso 1 usa più memoria


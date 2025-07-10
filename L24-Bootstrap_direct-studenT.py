import numpy as np, matplotlib.pyplot as plt

num_samples = 10000
sample_size = 10000
mu = 3
# CASO 2: t-Student diretta
scale = 0.4 #scala tstudent
df = 5  # gradi di libertà
sample_means2 = np.zeros(num_samples)

# Formula: sd_theoretical = scale * sqrt(df / (df - 2)), valida solo per df > 2
# È l'equivalente della sigma per la normale, ma per la t-Student dipende dai gradi di libertà
# Calcola deviazione standard teorica per t-Student
sd_theoretical = scale * np.sqrt(df / (df - 2))  # solo per df > 2

for i in range(num_samples):
    sample = mu + scale * np.random.standard_t(df, sample_size)
    sample_means2[i] = np.mean(sample)

mean_of_means2 = np.mean(sample_means2)
std_of_means2 = np.std(sample_means2)
theoretical_std2 = sd_theoretical / np.sqrt(sample_size)

print("\nCASO 2: Campionamento diretto da t-Student")
print("Media popolazione:", mu)
print("Media stimata:", mean_of_means2)
print("Sigma teorica (sigma/sqrt(N)):", theoretical_std2)
print("Deviazione standard delle medie:", std_of_means2)
plt.hist(sample_means2, bins=30)
plt.title("Caso 2: Distribuzione medie (t-Student)")
plt.show()
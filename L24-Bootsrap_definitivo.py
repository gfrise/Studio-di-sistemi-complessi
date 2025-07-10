import numpy as np, matplotlib.pyplot as plt

num_samples, sample_size, mu, sd = 10000, 10**5, 3, 0.4
sample_means = np.zeros(num_samples)

for i in range(num_samples):
    sample = np.random.normal(mu,sd,sample_size)
    sample_means[i] = np.mean(sample)

mean_of_means, std_of_means = np.mean(sample_means), np.std(sample_means)
theoritical_std = sd / np.sqrt(sample_size)

print("Media popolazione:", mu)
print("Media stimata:", mean_of_means)
print("Sigma teorica (sigma/sqrt(N)):", sd/np.sqrt(sample_size))
print("Deviazione standard delle medie:", std_of_means)
plt.hist(sample_means, bins=30)
plt.title("Distribuzione delle medie")
plt.show()







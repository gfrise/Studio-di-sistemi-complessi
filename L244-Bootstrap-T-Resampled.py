import numpy as np, matplotlib.pyplot as plt

# Parametri comuni
num_samples = 20000
sample_size = 20000
mu = 3

# CASO 4: Bootstrap da t-Student
scale = 0.4
df = 5
sample_means4 = np.zeros(num_samples)

# Genera un solo campione originale
original_sample = mu + scale * np.random.standard_t(df, sample_size)
original_mean = np.mean(original_sample)

for i in range(num_samples):
    # Campionamento con ripetizione dall'originale
    # Seleziona un campione di dimensione sample_size dall'array original_sample con ripetizione (bootstrap).
    bootstrap_sample = np.random.choice(original_sample, sample_size, replace=True)
    sample_means4[i] = np.mean(bootstrap_sample)

mean_of_means4 = np.mean(sample_means4)
std_of_means4 = np.std(sample_means4)
theoretical_std4 = np.std(original_sample) / np.sqrt(sample_size)

print("\nCASO 4: Bootstrap da t-Student")
print("Media popolazione:", mu)
print("Media campione originale:", original_mean)
print("Media stimata bootstrap:", mean_of_means4)
print("Sigma teorica (s/sqrt(N)):", theoretical_std4)
print("Deviazione standard delle medie bootstrap:", std_of_means4)
plt.hist(sample_means4, bins=30)
plt.title("Caso 4: Bootstrap medie (t-Student)")
plt.show()


# Stesse differenze del 3

# Forma distribuzione: Bootstrap eredita le code pesanti del campione originale
# sigma stimata dal campione, anche in 3

#Campione originale: [A, B, C]
# Con rimpiazzo: Possibili campioni bootstrap: [A, A, B], [B, C, C], [C, A, B], ecc.
# Senza rimpiazzo: Solo permutazioni: [A, B, C], [A, C, B], [B, A, C], ecc. (tutti contengono esattamente A,B,C una volta)
# Nel bootstrap, il campionamento con rimpiazzo è essenziale per stimare correttamente la variabilità e l'incertezza della statistica d'interesse (in questo caso la media).
# Perché il bootstrap usa SEMPRE con rimpiazzo?
# Simula il processo di campionamento reale dalla popolazione
# Permette variazioni tra i campioni bootstrap
# Alcune osservazioni appariranno più volte
# Alcune non appariranno affatto
# Genera incertezza simile a quella del campionamento originale
# rea campioni della stessa dimensione dell'originale

#CON RIPETIZIONE (replace=True)
# ✔️ Elementi possono ripetersi
# ✔️ Mantiene stessa dimensione campione originale
# ✔️ Alta variabilità tra campioni
# ✔️ Simula vero campionamento statistico
# ✔️ Usato in bootstrap per stimare incertezza
# SENZA RIPETIZIONE (replace=False)
# ✖️ Elementi unici (niente ripetizioni)
# ✖️ Dimensione ≤ originale
# ✖️ Bassa variabilità tra campioni
# ✖️ Solo permutazioni/divisioni
# ✖️ Usato in cross-validation, split dati
# PERCHÉ BOOTSTRAP USA SEMPRE CON RIPETIZIONE?
# 1️⃣ Riproduce l'incertezza del campionamento reale
# 2️⃣ Permette stime più robuste della variabilità
# 3️⃣ Genera campioni della stessa dimensione
# 4️⃣ Alcuni punti appaiono più volte (come in campionamento reale)
# 5️⃣ Senza ripetizione si avrebbero solo permutazioni (inutile per bootstrap)
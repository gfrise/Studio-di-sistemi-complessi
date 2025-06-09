import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
iterations, N = 1000, 10**5

def run_analysis(mu, scale, sampler, title):
    # 1) medie “popolazione”: generiamo un array (iterations × N) e facciamo la media riga-per-riga
    data = sampler(mu, scale, size=(iterations, N))
    pop_means = data.mean(axis=1)
    
    # 2) medie bootstrap: un unico campione empirico, poi indici casuali e media riga-per-riga
    emp = sampler(mu, scale, size=N)
    idx = np.random.randint(0, N, size=(iterations, N))
    boot_means = emp[idx].mean(axis=1)
    
    # 3) stampe delle deviazioni standard
    std_theoretical = scale / np.sqrt(N)
    print(f"\n--- {title} ---")
    print(f"Std teorica:   {std_theoretical:.6f}")
    print(f"Std pop_means: {pop_means.std():.6f}")
    print(f"Std bootstrap: {boot_means.std():.6f}")
    
    # 4) visualizzazione
    plt.figure(figsize=(12,3))
    plt.subplot(1,3,1)
    plt.hist(pop_means, bins=30, edgecolor='k', alpha=0.7)
    plt.title(f"{title}\nCampioni diretti")
    plt.subplot(1,3,2)
    plt.hist(boot_means, bins=30, edgecolor='k', alpha=0.7)
    plt.title(f"{title}\nBootstrap")
    plt.subplot(1,3,3)
    plt.scatter(pop_means, boot_means, alpha=0.3)
    m, M = pop_means.min(), pop_means.max()
    plt.plot([m,M],[m,M],'r--')
    plt.title("Confronto")
    plt.tight_layout()
    plt.show()

# distribuzione gaussiana
run_analysis(
    mu=3.0, 
    scale=0.4, 
    sampler=lambda mu, s, size: np.random.normal(loc=mu, scale=s, size=size),
    title="Gaussiana"
)

# distribuzione t-Student (df=5)
run_analysis(
    mu=3.0, 
    scale=0.4, 
    sampler=lambda mu, s, size: mu + s * np.random.standard_t(df=5, size=size),
    title="t-Student (df=5)"
)

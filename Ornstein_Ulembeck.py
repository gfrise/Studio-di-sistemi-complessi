import numpy as np
import matplotlib.pyplot as plt

# Compito 1: create_OU.c
# Create un codice che simuli un processo di Ornstein-Uhlenbeck a partire dalla sua equazione di Langevin. 
# Il codice deve anche prevedere la costruzione della funzione densità di probabilità (area normalizzata ad 1) 
# e della funzione di autocorrelazione. Fate in modo che i parametri rilevanti della simulazione siano inseriti dall’esterno, 
# per esempio tramite lettura da un file che contenga tutti i parametri. 
# Iterate il processo M = 100 volte e costruite l’istogramma e l’autocorrelazione mediati su queste M iterazioni,
# mostrando la standard deviation come barra d’errore.


# Compito 2: create_RISKEN.c

# Create un codice che simuli il processo multiscala definito dal seguente coefficiente di drift:

#     b(x) =
#         +k   se x > 0
#         -k   se x < 0

# Il processo è descritto dalla seguente equazione stocastica differenziale (SDE):

#     $$ dX_t = b(X_t) \, dt + \sigma \, dW_t $$

# Dove:
# - b(x) è definito a tratti come sopra,
# - σ (sigma) è l’intensità del rumore bianco,
# - W_t è un moto browniano standard.

# Richieste:
# 1. Implementare un’integrazione numerica del processo usando lo schema di Eulero–Maruyama.
# 2. Fare in modo che tutti i parametri rilevanti (k, σ, Δt, N, M, ecc.) siano letti da un file esterno
#    oppure passati da linea di comando.
#    - k: intensità del drift
#    - σ: intensità del rumore
#    - Δt: passo temporale
#    - N: numero di passi temporali per traiettoria
#    - M: numero di iterazioni (traiettorie da simulare)

# 3. Iterare la simulazione M volte e, per ciascuna:
#    - Generare una traiettoria del processo X(t)
#    - Salvare i dati delle traiettorie su file (opzionale)

# 4. Calcolare:
#    - L'istogramma della densità di probabilità stazionaria (area normalizzata a 1)
#    - La funzione di autocorrelazione del processo
#    - La media e la deviazione standard delle quantità sopra, calcolate su M traiettorie

# 5. Visualizzazione:
#    - Mostrare i risultati con barre d’errore corrispondenti alla deviazione standard

# Suggerimenti:
# - Usare strutture dati dinamiche o array allocati in modo efficiente per gestire i dati
# - Utilizzare una libreria esterna per la generazione di numeri casuali, se necessario
# - Separare il codice in funzioni modulari: lettura parametri, simulazione, analisi, output

tt = 10**4
step = 100
dt = 1/step
tauM = 50
tau = np.arange(0, tauM, dt)
nn = tt*step
m = 100

corr = 0

def autocorrelation(x):
    m1=0
    sd1=0
    
    for j in range(tt-tauM):
        m1 = m1 + x[j]
        sd1 = sd1 + x[j]**2
    
    m1 = m1 / (tt-tauM)
    sd1 = (sd1/(tt-tauM))-m1**2
    sd1 = sd1**0.5

    m2=0
    sd2=0
    corr=0
    for j in range(tt-tauM):
        m2 = m2+x[j+t]
        sd2 = sd2 + x[j+t]**2
        corr = corr + x[j]*x[j+t]

    m2 /= (tt-tauM)
    sd2 = (sd2/(tt-tauM))-m2**2
    sd2 = sd2**0.5
    corr /= (tt-tauM)
    return (corr-m1*m2)/(sd1*sd2)

print(corr)

#Ornstein e Ulembeck ha h(x) = -gamma*x e g(x)=c, poniamo c=1
gamma1 = 0.1
gamma2 = 0.2
med = np.zeros(tauM)
sd = np.zeros(tauM)

for k in range(m):
    x = np.zeros(nn)
    x[0] = 0.1
    noise = np.random.normal(0, np.sqrt(2), nn)
    for i in range (1,nn):
        x[i] = x[i-1] - gamma1*x[i-1]*dt + np.sqrt(dt)*noise[i]
    
    x_series = [x[t] for t in range(1, nn, step)]

    for t in range(tauM):
        ac = autocorrelation(x_series)
        med[t] += ac
        sd[t] += ac**2

for t in range(tauM):
    med[t] /= m
    sd[t] = sd[t]/m - med[t]**2
    print(f'{t}: {med[t]}\t{np.sqrt(sd[t])}')


x = np.linspace(0,tauM,50)
plt.semilogx(x, med, '.', c='black')
plt.errorbar(x, med, yerr=sd)
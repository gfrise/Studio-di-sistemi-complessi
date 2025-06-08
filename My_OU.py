import numpy as np
import matplotlib.pyplot as plt 

t, n, tauM, m, gamma = 10**4, 100, 50, 100, 0.1 #n:= punti per t unitario
dt, N, sum_ac, sum_ac2 = 1/n, n*t, np.zeros(tauM), np.zeros(tauM)

#    noise = np.random.randn()*np.sqrt(2*gamma*dt)
def OU():
    x = np.zeros(N)
    x[0]=0.1
    noise = np.random.normal(0,np.sqrt(2*dt),N) # o sqrt(2) o sqrt(dt)?
    for i in range(1,N):
        x[i]=x[i-1]-gamma*x[i-1]*dt+noise[i-1]
    return x

def AC(x,t):
    if t == 0:
        x1, x2 = x, x
    else :
        x1, x2 = x[:-t], x[t:]
    return (np.mean(x1*x2)-np.mean(x1)*np.mean(x2))/(np.std(x1)*np.std(x2))

for _ in range(m):
    x = OU()
    sample = x[::n]
    for t in range(tauM):
        ac = AC(sample,t)
        sum_ac[t] += ac
        sum_ac2[t] += ac*ac

means = sum_ac / m
variances = (sum_ac2 / m) - means**2
stds = np.sqrt(variances / m)

lags = np.arange(tauM)
plt.figure(figsize=(8,5))
plt.semilogy(lags, means, 'o', color='black', label='Autocorrelazione')
plt.errorbar(lags, means, yerr=stds, fmt='none', capsize=3, color='black')
plt.title("Autocorrelazione OU (media su {} traiettorie)".format(m))
plt.xlabel("Lag")
plt.ylabel("Autocorrelazione (scala log)")
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
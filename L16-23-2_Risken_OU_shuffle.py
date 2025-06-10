import numpy as np
import matplotlib.pyplot as plt
import random

h = lambda x : -2.*x
h1 = lambda x : -2.
g = lambda x : 1.

t, n, taum, m, gamma = 10**5, 100, 50, 100, 0.1 #n:= punti per step unitario
dt, N, ac, ac2 = 1/n, n*t, np.zeros(taum), np.zeros(taum)

def OU(val):
    x = np.zeros(N)
    x[0]=0.1
    noise = np.random.normal(0,np.sqrt(2*dt),N)
    dw = np.sqrt(2*dt)*np.random.randn(N)

    if val == 0 :
        for i in range(1,N):
            x[i]=x[i-1]-gamma*x[i-1]*dt+noise[i-1] 
            #x[i+1]=x[i]+h(x[i])*dt+g(x[i])*dw[i-1]
    elif val == 1 :
        for i in range(1,N):
            drift = gamma * dt if x[i-1] < 0 else -gamma * dt
            x[i] = x[i-1] + drift + noise[i-1]
    elif val == 2 :
        for i in range(1,N):
            xi = x[i-1]
            dx = h(xi)*dt+g(xi)*dw[i-1] + 0.5*h1(xi)*g(xi)*(dw[i-1]**2-2*dt)
            x[i]=xi+dx
    return x
    
def AC(x,t):
    if t==0:
        x1,x2 = x,x
    else:
        x1,x2=x[:-t],x[t:]
    return np.mean((x1*x2)-np.mean(x1)*np.mean(x2))/(np.std(x1)*np.std(x2))

def shuffle(x):
    for i in range(N):
        y = i + random.randint(0,N-i-1)#rand()%(N-i)
        x[i],x[y] = x[y], x[i]

####

def AC2(x):
    acf = np.zeros(taum)
    for lag in range(taum):
        cov = np.mean((x[:N-lag]-np.mean(x))*(x[lag:]-np.mean(x)))
        acf[lag]=cov/np.var(x)

def ensemble():
    mean_ens, mean2_ens = np.zeros(N),np.zeros(N)
    for _ in range(m):
        traj = OU(1)
        mean_ens[:]+=traj
        mean2_ens[:]+=traj**2
    mean_ens[:]/=m
    mean2_ens[:]/=m

###

for _ in range(m):
    x = OU(0)
    shuffle(x)
    sample = x[::n]
    for t in range(taum):
        a = AC(sample,t)
        ac[t] += a
        ac2[t] += a**2

means, lags = ac/m, np.arange(taum)
stds = np.sqrt((ac2/m - means**2)/m)

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
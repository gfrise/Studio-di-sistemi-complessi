import numpy as np
import matplotlib.pyplot as plt

h = lambda x: -2.*x
h1 = lambda x: -2.
g = lambda x: 1.

t, step, taum, m, y = 10**4, 100, 50, 100, 0.1 
dt, n, ac, ac2 = 1/step, step*t, np.empty(taum), np.empty(taum)  

def OU1():
    x = np.empty(n)
    x[0] = 0.1

    noise = np.random.normal(0,np.sqrt(2*dt),n)
    dw = np.sqrt(2*dt)*np.random.randn(n)

def OU2(n,sigma,dt,y):
    x = np.empty(n)
    x[0] = 0.1
    
    g1, g2 = np.random.normal(0,1,100), np.random.normal(0,1,100)
    z1 = sigma*np.sqrt(dt)*g1
    z2 = sigma*(g1/2 + g2/(2*np.sqrt(3)))*(dt**1.5)
    ydt = y*dt

    for i in range(1,n):
        x[i] = x[i-1]*(1-ydt+0.5*ydt**2) + z1[i-1] - y*z2[i-1] 
    
    return x



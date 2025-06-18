import numpy as np, matplotlib as plt

t, step, taum, m, y = 10**4, 100, 50, 100, 0.1
dt, n = 1/step, t*step

for i in range(m):
    x = np.empty(n)
    x[0] = 0.1
    noise = np.random.normal(0,np.sqrt(2),n)

    for j in range(1,n):
        x[j] = x[j-1] * (1 - y*dt) + np.sqrt(dt)*noise[j]

    x_set = [x[t] for t in range(1,n,step)]

    for t in range(taum):
        ac = 
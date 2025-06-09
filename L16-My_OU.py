import numpy as np
import matplotlib.pyplot as plt 

t, n, tauM, m, gamma = 10**3, 100, 50, 100, 0.1 #n:= punti per t unitario
dt, N, sum_ac, sum_ac2 = 1/n, n*t, np.zeros(tauM), np.zeros(tauM)

#   for(t=0;t<tmax;t++){
#      m2=0.;
#      sd2=0.;
#      corr=0.;
#      for(j=0; j<nR-tmax;j++){
# 	 m2=m2+X[j+t];
# 	 sd2=sd2+pow(X[j+t],2.);
# 	 corr=corr+X[j]*X[j+t];
#         }
# 	m2=m2/(double)(nR-tmax);
# 	sd2=(sd2/(double)(nR-tmax))-pow(m2,2.);
# 	sd2=pow(sd2,0.5);
# 	corr=corr/(double)(nR-tmax);
#         fprintf(fp3,"%d %lf \n", t, (corr-m1*m2)/(sd1*sd2)); 
#         }

# /*Eulero all'ordine dt^2*/
# 	x[0]=x0;
# 	for(i=0;i<N;i++){
# 	   x[i+1]=x[i]-gamma*x[i]*dt+Z1[i]-gamma*Z2[i]+0.5*pow(gamma,2.)*x[i]*pow(dt,2.);
# 	}
	
#       double *Z1,*Z2,*Z3;
#   Z1=allocamem1(nR*enne);
#   Z2=allocamem1(nR*enne);
#   Z3=allocamem1(nR*enne);
#   for(i=0; i<nR*enne;i++){
#       Z1[i]=GAMMA[i]*sqrt(delt);
#       Z2[i]=(GAMMA[i]/2.+GAMMA2[i]/(2.*sqrt(3.)))*pow(delt,1.50);
#       Z3[i]=(pow(GAMMA[i],2.)+GAMMA3[i]+0.50)*pow(delt,2.)/3.;
#     }
#   free(GAMMA);
#   free(GAMMA2);
#   free(GAMMA3);

#   /*FILE *fp6;
#   fp6=fopen("serienumericagauss.dat", "w");
#   for(i=0; i<nR*enne;i++){
#      fprintf(fp6,"%lf \n",GAMMA[i]);
#   }
#   fclose(fp6);*/
	  
#   double *delXs,*delX2s,*delX3s,*delX4s,*Xs;
#   delXs=allocamem1(nR*enne);
#   delX2s=allocamem1(nR*enne);
#   delX3s=allocamem1(nR*enne);
#   delX4s=allocamem1(nR*enne);
#   Xs=allocamem1(nR*enne);

#   Xs[0]=x0;
#   for(i=0; i<nR*enne;i++){
#          delXs[i]=-(gammaOU*Xs[i-1])*delt+Z1[i];
#          delX2s[i]=-gammaOU*Z2[i];
#          delX3s[i]=0.0*Z3[i];
#          delX4s[i]=0.5*pow(gammaOU,2.)*Xs[i-1]*pow(delt,2.0);
#       /*delXs[i]=-0.1*Xs[i-1]*delt+GAMMA[i]*sqrt(delt);*/
#       Xs[i]=Xs[i-1]+delXs[i]+delX2s[i]+delX3s[i]+delX4s[i];
#   }


#    noise = np.random.randn()*np.sqrt(2*gamma*dt)
def OU():
    x = np.zeros(N)
    x[0]=0.1
    noise = np.random.normal(0,np.sqrt(2*dt),N) # o sqrt(2) o sqrt(dt)?
    for i in range(1,N):
        x[i]=x[i-1]-gamma*x[i-1]*dt+noise[i-1]
    return x

# def Risken():
#     x = np.zeros(N + 1)
#     x[0] = 0.1
#     noise = np.random.normal(0,np.sqrt(2*dt),N)
#     for i in range(N):
#         drift = gamma * dt if x[i] < 0 else -gamma * dt
#         x[i + 1] = x[i] + drift + noise[i]
#     return x

# acf = np.zeros(tmax)
# for lag in range(tmax):
#     cov = np.mean((y[:T-lag] - mean_y) * (y[lag:] - mean_y))
#     acf[lag] = cov / var_y

# Ensemble Monte Carlo
# def run_ensemble():
#     for _ in range(M):
#         traj = simulate_ou_euler(lam, sigma, dt, N, x0)
#         mean_ensemble[:] += traj
#         mean_sq_ensemble[:] += traj**2
#     mean_ensemble[:] /= M
#     mean_sq_ensemble[:] /= M

# # Esecuzione
# run_ensemble()

# time = np.linspace(0, T, N)
# steady_var = sigma**2 / (-2.0 * lam)

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

#AGGIUNGERE MEDIA ENSEMBLE
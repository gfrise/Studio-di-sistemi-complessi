#include<stdio.h>
#include<stdlib.h>

int main(){
  for(t=0;t<tmax;t++){
     m2=0.;
     sd2=0.;
     corr=0.;
     for(j=0; j<nR-tmax;j++){
	 m2=m2+X[j+t];
	 sd2=sd2+pow(X[j+t],2.);
	 corr=corr+X[j]*X[j+t];
        }
	m2=m2/(double)(nR-tmax);
	sd2=(sd2/(double)(nR-tmax))-pow(m2,2.);
	sd2=pow(sd2,0.5);
	corr=corr/(double)(nR-tmax);
        fprintf(fp3,"%d %lf \n", t, (corr-m1*m2)/(sd1*sd2)); 
        }

/*Eulero all'ordine dt^2*/
	x[0]=x0;
	for(i=0;i<N;i++){
	   x[i+1]=x[i]-gamma*x[i]*dt+Z1[i]-gamma*Z2[i]+0.5*pow(gamma,2.)*x[i]*pow(dt,2.);
	}
	
      double *Z1,*Z2,*Z3;
  Z1=allocamem1(nR*enne);
  Z2=allocamem1(nR*enne);
  Z3=allocamem1(nR*enne);
  for(i=0; i<nR*enne;i++){
      Z1[i]=GAMMA[i]*sqrt(delt);
      Z2[i]=(GAMMA[i]/2.+GAMMA2[i]/(2.*sqrt(3.)))*pow(delt,1.50);
      Z3[i]=(pow(GAMMA[i],2.)+GAMMA3[i]+0.50)*pow(delt,2.)/3.;
    }
  free(GAMMA);
  free(GAMMA2);
  free(GAMMA3);

  FILE *fp6;
  fp6=fopen("serienumericagauss.dat", "w");
  for(i=0; i<nR*enne;i++){
     fprintf(fp6,"%lf \n",GAMMA[i]);
  }
  fclose(fp6);
	  
  double *delXs,*delX2s,*delX3s,*delX4s,*Xs;
  delXs=allocamem1(nR*enne);
  delX2s=allocamem1(nR*enne);
  delX3s=allocamem1(nR*enne);
  delX4s=allocamem1(nR*enne);
  Xs=allocamem1(nR*enne);

  Xs[0]=x0;
  for(i=0; i<nR*enne;i++){
         delXs[i]=-(gammaOU*Xs[i-1])*delt+Z1[i];
         delX2s[i]=-gammaOU*Z2[i];
         delX3s[i]=0.0*Z3[i];
         delX4s[i]=0.5*pow(gammaOU,2.)*Xs[i-1]*pow(delt,2.0);
      /*delXs[i]=-0.1*Xs[i-1]*delt+GAMMA[i]*sqrt(delt);*/
      Xs[i]=Xs[i-1]+delXs[i]+delX2s[i]+delX3s[i]+delX4s[i];
  }

}

///
 m1=0.;
  sd1=0.;
  for(j=0; j<nR-tmax;j++){
     m1=m1+X[j];
     sd1=sd1+pow(X[j],2.);
  }
  m1=m1/(double)(nR-tmax);
  sd1=(sd1/(double)(nR-tmax))-pow(m1,2.);
  sd1=pow(sd1,0.5);
  
  for(t=0;t<tmax;t++){
     m2=0.;
     sd2=0.;
     corr=0.;
     for(j=0; j<nR-tmax;j++){
	 m2=m2+X[j+t];
	 sd2=sd2+pow(X[j+t],2.);
	 corr=corr+X[j]*X[j+t];
        }
	m2=m2/(double)(nR-tmax);
	sd2=(sd2/(double)(nR-tmax))-pow(m2,2.);
	sd2=pow(sd2,0.5);
	corr=corr/(double)(nR-tmax);
        fprintf(fp3,"%d %lf \n", t, (corr-m1*m2)/(sd1*sd2)); 
        }
  fclose(fp3);  
  
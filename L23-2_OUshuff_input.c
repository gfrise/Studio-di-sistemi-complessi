#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "myClibrary.h"

#define PIgreco 3.1415926535897932385


int main(int argc, char* argint[]) {

  int i,j;

  /*printf("Ho ricevuto %d argomenti\n", argc-1);
  printf("Questi argomenti sono:\n");
  
  for(i=1; i<=argc-1; i++)
    printf("%s\n", argint[i]);*/


  FILE *fp0;
  double param[6];
  fp0=fopen("input_parametersOU.dat", "r");
  for(i=0; i<6;i++){
     fscanf(fp0,"%lf \n",&param[i]);
  }
  fclose(fp0);

  printf(" \n");
  double gammaOU;
  printf("Valore del parametro gamma \n");
  gammaOU=param[0];
  printf("gamma= %lf \n",gammaOU);
	  
  printf("\n");
  double delt;
  int enne;
  printf("Numero di punti per unita' di tempo \n");
  enne=(int)param[1];
  delt=pow(enne,-1.);
  printf("Il time-step e': %lf \n", delt);
  
  
  printf("\n");
  double x0;
  printf("Starting-point della simulazione \n");
  x0=param[2];
  printf("x0= %lf \n",x0);
  

  printf("\n");
  int nR;
  printf("Numero di punti che vuoi generare \n");
  nR=(int)param[3];
  printf("nR= %d \n",nR);
  
  printf("\n");
  int seed;
  printf("Seme della simulazione \n");
  seed=atoi(argint[1]);
  printf("%d \n",seed);

  printf("\n");
  int seedsh;
  printf("Seme dello Shuffling \n");
  seed=atoi(argint[2]);
  printf("%d \n",seedsh);

  printf("\n Inizio della simulazione \n");
  double *GAMMA;
  GAMMA=allocamem1(nR*enne);
  double x1,x2,gam;
  double m=0;
  double s=2;
  srand(seed);
  for(i=0; i<nR*enne;i++){
         x1=(float)rand()/(float)RAND_MAX;
	  if(x1>0){
	     x2=(float)rand()/(float)RAND_MAX;
	     gam=sqrt(-2.*log(x1))*sin(2.*PIgreco*x2);
	     GAMMA[i]=(double) sqrt(s)*gam;
	     GAMMA[i]=m+GAMMA[i];
	  }
    }

  double *Z1;
  Z1=allocamem1(nR*enne);
  for(i=0; i<nR*enne;i++){
      Z1[i]=GAMMA[i]*sqrt(delt);
    }

  /*FILE *fp6;
  fp6=fopen("serienumericagauss.dat", "w");
  for(i=0; i<nR*enne;i++){
     fprintf(fp6,"%lf \n",GAMMA[i]);
  }
  fclose(fp6);*/
	  
  double *delXs,*Xs;
  delXs=allocamem1(nR*enne);
  Xs=allocamem1(nR*enne);

  Xs[0]=x0;
  for(i=0; i<nR*enne;i++){
      delXs[i]=-(gammaOU*Xs[i-1])*delt+Z1[i];
      Xs[i]=Xs[i-1]+delXs[i];
  }

  double *X;
  int p;
  X=allocamem1(nR);
  for(i=0; i<nR*enne;i++){
     p=(double) i/enne;
     X[p]=Xs[i];
  }

  int pos;
  double temp;
  srand(seedsh);
  for(i=0;i<nR;i++){
	    pos=i+rand()%(nR-i);
	    
	    temp=X[i];
	    X[i]=X[pos];
	    X[pos]=temp;
  }

  /*FILE *fp5;
  fp5=fopen("serienumericaOU.dat", "w");
  for(i=0; i<nR;i++){
     fprintf(fp5,"%lf \n",X[i]);
  }
  fclose(fp5);*/ 
  printf("\n Fine della simulazione \n");
  free(delXs);
  free(Z1);
  free(GAMMA);
  free(Xs);



  /*Statistica di base*/
  printf(" \n");
  double emme,std,minn,maxx;
  emme=mymedia(nR,X);
  printf("la media e' %lf \n", emme);
  std=mysd(nR,X);
  printf("la standard deviation e' %lf \n", std);
  minn=mymin(nR,X);
  printf("il minimo e' %lf \n", minn);
  maxx=mymax(nR,X);
  printf("il massimo e' %lf \n", maxx);
  
  
  /*istogramma*/
  int nbin;
  printf(" \n");
  printf("Numero di bin da usare per l'istogramma \n");
  nbin=(int)param[4];
  printf("%d \n",nbin);
  
  istogramma(nR,X,minn,maxx,nbin);
  
  /*auto-correlation */
  int t,tmax;
  printf(" \n");
  printf("Tmax dell'autocorrelazione \n");
  tmax=(int) param[5];
  printf("%d \n",tmax);

  FILE *fp3;
  fp3=fopen("auto-correlationOU.dat","w");

  double corr,m1,m2,sd1,sd2;
  
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
  
  
  
  
  
  
}

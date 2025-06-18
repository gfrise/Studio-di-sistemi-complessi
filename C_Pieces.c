#include<stdio.h>
#include<stdlib.h>


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
  
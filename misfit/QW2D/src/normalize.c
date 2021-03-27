/* Data positive transform for OT 



*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "normalize.h"

double normalize(double *fn, double *f, double *gn, double *g, int n, int flag)
{
	int i;
	double sum1, sum2;
	sum1 = 0.;
	sum2 = 0.;
	for(i=0; i<n; i++){
		sum1 += f[i];
		sum2 += g[i];
	}
	if(flag){
		sum1 /= (double)n;
		sum2 /= (double)n;
	}
	for(i=0; i<n; i++){
		if(sum1 > 0)
			fn[i] = f[i]/sum1;
		else 
			fn[i] = 0.;
		if(sum2 > 0)
			gn[i] = g[i]/sum2;
		else
			gn[i] = 0.;
	}
	return sum1;
}
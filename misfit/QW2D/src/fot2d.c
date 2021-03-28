#include "fot2d.h"

/* poisson solver */
float *create_negative_laplace_kernel2d(int n1, int n2){
	float *kernel=(float *)calloc(n1*n2,sizeof(float));
	int i, j;
	float x, y, negativeLaplacian;
	for(i=0;i<n2;i++){
		for(j=0;j<n1;j++){
			x=M_PI*j/(n1*1.0);
			y=M_PI*i/(n2*1.0);    
			negativeLaplacian=2*n1*n1*(1-cos(x))+2*n2*n2*(1-cos(y));
			kernel[i*n1+j]=negativeLaplacian; 
		}
	}
	return kernel;
}

struct poisson_solver create_poisson_solver_workspace2d(int n1, int n2){
	struct poisson_solver fftps;
	fftps.workspace=(float *)calloc(n1*n2,sizeof(float));
	fftps.kernel=create_negative_laplace_kernel2d(n1,n2);

// #ifdef _OPENMP
// 	fftw_plan_with_nthreads(omp_get_max_threads());
// #endif	
	fftps.dctIn=fftwf_plan_r2r_2d(n2, n1, fftps.workspace, fftps.workspace,
								FFTW_REDFT10, FFTW_REDFT10,
								FFTW_MEASURE);
	fftps.dctOut=fftwf_plan_r2r_2d(n2, n1, fftps.workspace, fftps.workspace,
								FFTW_REDFT01, FFTW_REDFT01,
								FFTW_MEASURE);  

	if(fftps.dctIn == NULL){
		printf("failed to construct fftw plan\n");
		exit(1);
	}
	return fftps;
}

void destroy_poisson_solver(struct poisson_solver fftps){
	free(fftps.kernel);
	free(fftps.workspace);
	fftwf_destroy_plan(fftps.dctIn);
	fftwf_destroy_plan(fftps.dctOut);
}


/* convex conjugate */
void alloc_hull(struct convex_hull *hull, int n){
	(*hull).indices=(int *)calloc(n, sizeof(int));
	(*hull).hullCount=0; 
}

void init_hull(struct convex_hull *hull, int n){
	memset(hull->indices, 0, n*sizeof(int));
	hull->hullCount = 0;
}


void destroy_hull(struct convex_hull *hull){
	free(hull->indices);
}

void add_point(float *u, struct convex_hull *hull, int i){
	int hc, ic1, ic2;
	float oldSlope, slope;
	if(hull->hullCount<2){
		hull->indices[1]=i;
		hull->hullCount++;
	}else{
		hc=hull->hullCount;
		ic1=hull->indices[hc-1];
		ic2=hull->indices[hc-2];

		oldSlope=(float)((u[ic1]-u[ic2])/(ic1-ic2));
		slope=(float)((u[i]-u[ic1])/(i-ic1));

		if(slope>=oldSlope){
			hc=hull->hullCount;
			hull->indices[hc]=i;
			hull->hullCount++;
		}else{
			hull->hullCount--;
			add_point(u, hull, i);
		}
	}
}

void get_convex_hull(float *u, struct convex_hull *hull, int n){
	int i;
	hull->indices[0]=0;
	hull->indices[1]=1;
	hull->hullCount=2;
	for(i=2;i<n;i++){   
		add_point(u, hull, i);     
	}
}

void compute_dual_indicies(int *dualIndicies, float *u, struct convex_hull *hull, int n){

	int counter=1;
	int hc=hull->hullCount;
	int i, ic1, ic2;
	float s, slope;
	for(i=0;i<n;i++){    
		s=(i+.5)/(n*1.0);
		ic1=hull->indices[counter];
		ic2=hull->indices[counter-1];

		slope=(float)(n*(u[ic1]-u[ic2])/(ic1-ic2));
		while(s>slope&&counter<hc-1){
			counter++;
			ic1=hull->indices[counter];
			ic2=hull->indices[counter-1];
			slope=(float)(n*(u[ic1]-u[ic2])/(ic1-ic2));
		}
		dualIndicies[i]=hull->indices[counter-1];   
	}
}

void compute_dual(float *dual, float *u, int *dualIndicies, struct convex_hull *hull, int n){
	int i, index;
	float s, x, v1, v2;
	get_convex_hull(u, hull, n);
	compute_dual_indicies(dualIndicies, u, hull, n);
	for(i=0;i<n;i++){
		s=(i+.5)/(n*1.0);
		index=dualIndicies[i];
		x=(index+.5)/(n*1.0);
		v1=s*x-u[dualIndicies[i]];
		v2=s*(n-.5)/(n*1.0)-u[n-1];
		if(v1>v2){
			dual[i]=v1;
		}else{
			dualIndicies[i]=n-1;
			dual[i]=v2;
		}
	}
}

void transpose_floats(float *transpose, float *data, int n1, int n2){  
	int i, j;
	for(i=0;i<n2;i++){
		for(j=0;j<n1;j++){     
			transpose[j*n2+i]=data[i*n1+j];
		}
	}
}

void compute_2d_dual(float *dual, float *u, struct convex_hull *hull, int n1, int n2){
	int pcount=n1*n2;
	int n=fmax(n1,n2);
	int *argmin=(int *)calloc(n,sizeof(int));
	float *temp=(float *)calloc(pcount,sizeof(float));
	memcpy(temp, u, pcount*sizeof(float));

	int i, j;
	for(i=0;i<n2;i++){
		compute_dual(&dual[i*n1], &temp[i*n1], argmin, hull, n1);
	}
	transpose_floats(temp, dual, n1, n2);
	for(i=0;i<n1*n2;i++){
		dual[i]=-temp[i];
	}
	for(j=0;j<n1;j++){
		compute_dual(&temp[j*n2], &dual[j*n2], argmin, hull, n2);   
	}
	transpose_floats(dual, temp, n2, n1);

	free(temp);
	free(argmin);
}
void convexify(float *phi, float *dual, struct convex_hull *hull, int n1, int n2){
	compute_2d_dual(dual, phi, hull, n1, n2);

	compute_2d_dual(phi, dual, hull, n1, n2);
}

/* FOT Kernels */
void alloc_fotSpace_2d(struct fotSpace *fotspace, int n1, int n2){
	// 2D array in size [n2, n1], which is in row-major.
	int n = fmax(n1, n2);
	(*fotspace).fftps = create_poisson_solver_workspace2d(n1, n2);
	(*fotspace).xMap = (float*)calloc((n1+1)*(n2+1), sizeof(float));
	(*fotspace).yMap = (float*)calloc((n1+1)*(n2+1), sizeof(float));
	(*fotspace).f = (float*)calloc(n1*n2, sizeof(float));
	(*fotspace).g = (float*)calloc(n1*n2, sizeof(float));
	(*fotspace).mu = (float*)calloc(n1*n2, sizeof(float));
	(*fotspace).nu = (float*)calloc(n1*n2, sizeof(float));
	(*fotspace).phi = (float*)calloc(n1*n2, sizeof(float));
	(*fotspace).dual = (float*)calloc(n1*n2, sizeof(float));
	(*fotspace).rho = (float*)calloc(n1*n2, sizeof(float));
	(*fotspace).wd = (float*)calloc((*fotspace).nIter, sizeof(float));
	(*fotspace).gn = (float*)calloc((*fotspace).nIter, sizeof(float));
	alloc_hull(&((*fotspace).hull), n);
}


void init_map(float* xMap, float* yMap, int n1, int n2){
	int i, j;
	float x, y;
	for(i=0; i<n2+1; i++){
		for(j=0; j<n1+1; j++){
			x = j/(n1*1.0);
			y = i/(n2*1.0);
			xMap[i*n1+j] = x;
			yMap[i*n1+j] = y;
		}
	}
}

void init_phi_dual(float* phi, float *dual, int n1, int n2){
	int i, j;
	float x, y, z;
	for(i=0; i<n2; i++){
		for(j=0; j<n1; j++){
			x = (j+.5)/(n1*1.0);
			y = (i+.5)/(n2*1.0);
			z = .5*(x*x + y*y);
			phi[i*n1+j] = z;
			dual[i*n1+j] = z;
		}
	}
}

float init_step_size(float* mu, float* nu, int n){
	int i;
	float max1, max2;
	max1 = max2 = .0;
	for(i=0; i<n; i++){
		if(mu[i]>max1) max1 = mu[i];
		if(nu[i]>max2) max2 = nu[i];
	}
	return fmax(max1, max2); // the step length will affect the results
}

void init_fotSpace_2d(struct fotSpace* fotspace, int n1, int n2, float* signal1, float* signal2){
	init_map((*fotspace).xMap, (*fotspace).yMap, n1, n2);
	init_phi_dual((*fotspace).phi, (*fotspace).dual, n1, n2);
	int n = fmax(n1, n2);
	init_hull(&((*fotspace).hull), n);
	memcpy((*fotspace).f, signal1, n1*n2*sizeof(float));
	memcpy((*fotspace).g, signal2, n1*n2*sizeof(float));
	memset((*fotspace).wd, 0, (*fotspace).nIter*sizeof(float));
	memset((*fotspace).gn, 0, (*fotspace).nIter*sizeof(float));
}

void destroy_fotSpace_2d(struct fotSpace *fotspace){
	destroy_hull(&((*fotspace).hull));
	free((*fotspace).xMap);
	free((*fotspace).yMap);
	free((*fotspace).rho);
	free((*fotspace).phi);
	free((*fotspace).dual);
	free((*fotspace).mu);
	free((*fotspace).nu);
	free((*fotspace).f);
	free((*fotspace).g);
	destroy_poisson_solver((*fotspace).fftps);
}

int sgn(float x){   
	int truth=(x>0)-(x<0);
	return truth;
}

float interpolate_function(float *function, float x, float y, int n1, int n2){
	int xIndex, yIndex, xOther, yOther;
	float xfrac, yfrac, v1, v2, v3, v4, v;

	xIndex=fmin(fmax(x*n1-.5 ,0),n1-1);
	yIndex=fmin(fmax(y*n2-.5 ,0),n2-1);

	xfrac=x*n1-xIndex-.5;
	yfrac=y*n2-yIndex-.5;

	xOther=xIndex+sgn(xfrac); 
	yOther=yIndex+sgn(yfrac);

	xOther=fmax(fmin(xOther, n1-1),0);
	yOther=fmax(fmin(yOther, n2-1),0);

	v1=(1-fabs(xfrac))*(1-fabs(yfrac))*function[yIndex*n1+xIndex];
	v2=fabs(xfrac)*(1-fabs(yfrac))*function[yIndex*n1+xOther];
	v3=(1-fabs(xfrac))*fabs(yfrac)*function[yOther*n1+xIndex];
	v4=fabs(xfrac)*fabs(yfrac)*function[yOther*n1+xOther];

	v=v1+v2+v3+v4;

	return v;  
}

void calc_pushforward_map(float *xMap, float *yMap, float *dual, int n1, int n2){
    
	float xStep=1.0/n1;
	float yStep=1.0/n2;
	int i, j;
	float x, y, dualxp, dualxm, dualyp, dualym;
#ifdef _OPENMP
#pragma omp parallel default(shared) private(i,j,x,y,dualxp,dualxm,dualyp,dualym)
#endif
{
#ifdef _OPENMP
#pragma omp for schedule(guided) nowait
#endif	
	for(i=0;i<n2+1;i++){
		for(j=0;j<n1+1;j++){
			x=j/(n1*1.0);
			y=i/(n2*1.0);

			dualxp=interpolate_function(dual, x+xStep, y, n1, n2);
			dualxm=interpolate_function(dual, x-xStep, y, n1, n2);

			dualyp=interpolate_function(dual, x, y+yStep, n1, n2);
			dualym=interpolate_function(dual, x, y-yStep, n1, n2);

			xMap[i*(n1+1)+j]=.5*n1*(dualxp-dualxm);
			yMap[i*(n1+1)+j]=.5*n2*(dualyp-dualym);       
		}
	}

}

}
/*
void calc_pushforward_map_gsl(float *xMap, float *yMap, float *dual, int n1, int n2)
{
	int i, j;
	const gsl_interp2d_type *T = gsl_interp2d_bilinear;
	gsl_interp2d *spline = gsl_interp2d_alloc(T, n1, n2);
	gsl_interp_accel *xacc = gsl_interp_accel_alloc();
	gsl_interp_accel *yacc = gsl_interp_accel_alloc();

	float* xa = (float*)malloc(n1*sizeof(float));
	float* ya = (float*)malloc(n2*sizeof(float));
	for(i=0; i<n1; i++)
		xa[i] = (i+.5)/n1;
	for(i=0; i<n2; i++)
		ya[i] = (i+.5)/n2;
	// initialize interpolation 
	gsl_interp2d_init(spline, xa, ya, dual, n1, n2);

	float xStep=1.0/n1;
	float yStep=1.0/n2;
	float x, y, dualxp, dualxm, dualyp, dualym;    
	// interpolate values 
#ifdef _OPENMP
#pragma omp parallel default(shared) private(i,j,x,y,dualxp,dualxm,dualyp,dualym)
#endif
{
#ifdef _OPENMP
#pragma omp for schedule(guided) nowait
#endif	
	for(i=0; i<n2+1; i++){
		for(j=0; j<n1+1; j++){
			x=j/(n1*1.0);
			y=i/(n2*1.0);

			dualxp = gsl_interp2d_eval_extrap(spline, xa, ya, dual, x+xStep, y, xacc, yacc);
			dualxm = gsl_interp2d_eval_extrap(spline, xa, ya, dual, x-xStep, y, xacc, yacc);
			dualyp = gsl_interp2d_eval_extrap(spline, xa, ya, dual, x, y+yStep, xacc, yacc);
			dualym = gsl_interp2d_eval_extrap(spline, xa, ya, dual, x, y-yStep, xacc, yacc);

			xMap[i*(n1+1)+j]=.5*n1*(dualxp-dualxm);
			yMap[i*(n1+1)+j]=.5*n2*(dualyp-dualym);
		}
	}
}

}
*/
void sampling_pushforward(float *rho, float *mu, float *xMap, float *yMap, int n1, int n2, float totalMass){
	
	int pcount=n1*n2;
	memset(rho,0,pcount*sizeof(float));

	float xCut=pow(1.0/n1,1.0/3);
	float yCut=pow(1.0/n2,1.0/3);
	int i, j, l, k;
	int xIndex, yIndex, xOther, yOther, xSamples, ySamples;
	float xStretch0, yStretch0, xStretch1, yStretch1, xStretch, yStretch, mass;
	float xFrac, yFrac, factor, xPoint, yPoint, X, Y, a, b;
#ifdef _OPENMP
#pragma omp parallel default(shared) private(i,j,l,k,mass,xStretch0,xStretch1,xStretch,yStretch0,yStretch1,yStretch, \
										xSamples,ySamples,factor,a,b,xPoint,yPoint,X,Y,xIndex,yIndex,xFrac,yFrac,xOther,yOther)
#endif
{
#ifdef _OPENMP
#pragma omp for schedule(guided) nowait
#endif
	for(i=0;i<n2;i++){
		for(j=0;j<n1;j++){
			mass=mu[i*n1+j];
			if(mass>0){        
				xStretch0=fabs(xMap[i*(n1+1)+j+1]-xMap[i*(n1+1)+j]);
				xStretch1=fabs(xMap[(i+1)*(n1+1)+j+1]-xMap[(i+1)*(n1+1)+j]);

				yStretch0=fabs(yMap[(i+1)*(n1+1)+j]-yMap[i*(n1+1)+j]);
				yStretch1=fabs(yMap[(i+1)*(n1+1)+j+1]-yMap[i*(n1+1)+j+1]);

				xStretch=fmax(xStretch0, xStretch1);
				yStretch=fmax(yStretch0, yStretch1);

				xSamples=2*fmax(n1*xStretch,1);
				ySamples=2*fmax(n2*yStretch,1);

				if(xStretch<xCut&&yStretch<yCut){			    
					factor=1/(xSamples*ySamples*1.0);
					for(l=0;l<ySamples;l++){
						for(k=0;k<xSamples;k++){

							a=(k+.5)/(xSamples*1.0);
							b=(l+.5)/(ySamples*1.0);

							xPoint=(1-b)*(1-a)*xMap[i*(n1+1)+j]+(1-b)*a*xMap[i*(n1+1)+j+1]+b*(1-a)*xMap[(i+1)*(n1+1)+j]+a*b*xMap[(i+1)*(n1+1)+j+1];
							yPoint=(1-b)*(1-a)*yMap[i*(n1+1)+j]+(1-b)*a*yMap[i*(n1+1)+j+1]+b*(1-a)*yMap[(i+1)*(n1+1)+j]+a*b*yMap[(i+1)*(n1+1)+j+1];

							X=xPoint*n1-.5;
							Y=yPoint*n2-.5;

							xIndex=(int)X;
							yIndex=(int)Y;

							xFrac=X-xIndex;
							yFrac=Y-yIndex;

							xOther=xIndex+1;
							yOther=yIndex+1;

							xIndex=fmin(fmax(xIndex,0),n1-1);
							xOther=fmin(fmax(xOther,0),n1-1);

							yIndex=fmin(fmax(yIndex,0),n2-1);
							yOther=fmin(fmax(yOther,0),n2-1);
							                     
							rho[yIndex*n1+xIndex]+=(1-xFrac)*(1-yFrac)*mass*factor;
							rho[yOther*n1+xIndex]+=(1-xFrac)*yFrac*mass*factor;
							rho[yIndex*n1+xOther]+=xFrac*(1-yFrac)*mass*factor;
							rho[yOther*n1+xOther]+=xFrac*yFrac*mass*factor;

						}
					}
				}
			}  
		}
	}
}
	float sum=0;
	for(i=0;i<pcount;i++){
		sum+=rho[i]/pcount;
	}
	for(i=0;i<pcount;i++){
		rho[i]*=totalMass/sum;
	}

}

float update_potential(struct poisson_solver fftps, float *phi, float *rho, float *nu, float sigma, int pcount){
	float h1=0;
	int i;
	for(i=0;i<pcount;i++){
		fftps.workspace[i]=(rho[i]-nu[i]);
	}

	fftwf_execute(fftps.dctIn);

	fftps.workspace[0]=0;
	for(i=1;i<pcount;i++){
		fftps.workspace[i]/=4*pcount*fftps.kernel[i];      
	}

	fftwf_execute(fftps.dctOut);

	for(i=0;i<pcount;i++){
		phi[i]+=sigma*fftps.workspace[i];
		h1+=fftps.workspace[i]*(rho[i]-nu[i]);
	}

	h1/=pcount;
	return h1;
}

float step_update(float sigma, float value, float oldValue, float gradSq, float scaleUp, float scaleDown, float upper, float lower){
    
	float diff=value-oldValue;

	if(diff>gradSq*sigma*upper){
		return sigma*scaleUp;
	}else if(diff<gradSq*sigma*lower){
		return sigma*scaleDown;
	}else{
		return sigma;
	}
    
}

float compute_w2(float *phi, float *dual, float *mu, float *nu, int n1, int n2){   
	int pcount=n1*n2; 
	float value=0.;
	float x, y;
	int i, j;  
	for(i=0;i<n2;i++){
		for(j=0;j<n1;j++){
			x=(j+.5)/(n1*1.0);
			y=(i+.5)/(n2*1.0);
			value+=.5*(x*x+y*y)*(mu[i*n1+j]+nu[i*n1+j])-nu[i*n1+j]*phi[i*n1+j]-mu[i*n1+j]*dual[i*n1+j];
		}
	}
	value/=pcount;
	return value; 
}

float compute_l2_fot2d(float *mu, float *nu, float *phi, float *dual, float *rho,  
		float *xMap, float *yMap, float totalMass, struct poisson_solver fftps, struct convex_hull hull, 
		float sigma, int maxIters, int n1, int n2, float *values, float *grad_norms, int verbose){
	int i, j;
	int pcount = n1*n2;

	// initialization
	memcpy(rho, mu, pcount*sizeof(float));

	float oldValue, value, scaleDown, scaleUp, upper, lower, gradSq, x, y;
	scaleDown = .8;
	scaleUp = 1./scaleDown;
	upper = .75;
	lower = .25;
	oldValue = compute_w2(phi, dual, mu, nu, n1, n2);

	clock_t time1, time2;
	float time_update_potential = 0;
	float time_convexify = 0;
	float time_pushforward = 0;
	float time_calc_map = 0;
	for(i=0; i<maxIters; i++){
		time1 = clock(); 
		gradSq = update_potential(fftps, phi, rho, nu, sigma, pcount);
		time2 = clock();
		time_update_potential += (time2 - time1)/(CLOCKS_PER_SEC*1.0);	

		time1 = clock();	
		convexify(phi, dual, &hull, n1, n2);
		time2 = clock();
		time_convexify += (time2 - time1)/(CLOCKS_PER_SEC*1.0);

		value = compute_w2(phi, dual, mu, nu, n1, n2);
		sigma = step_update(sigma, value, oldValue, gradSq, scaleUp, scaleDown, upper, lower);
		oldValue = value;

		time1 = clock();
		calc_pushforward_map(xMap, yMap, phi, n1, n2);
		time2 = clock();
		time_calc_map += (time2 - time1)/(CLOCKS_PER_SEC*1.0);

		time1 = clock();
		sampling_pushforward(rho, nu, xMap, yMap, n1, n2, totalMass);
		time2 = clock();
		time_pushforward += (time2 - time1)/(CLOCKS_PER_SEC*1.0);

		time1 = clock(); 
		gradSq = update_potential(fftps, dual, rho, mu, sigma, pcount);
		time2 = clock();
		time_update_potential += (time2 - time1)/(CLOCKS_PER_SEC*1.0);	

		time1 = clock();		
		convexify(dual, phi, &hull, n1, n2);
		time2 = clock();
		time_convexify += (time2 - time1)/(CLOCKS_PER_SEC*1.0);

		time1 = clock();		
		calc_pushforward_map(xMap, yMap, dual, n1, n2);
		time2 = clock();
		time_calc_map += (time2 - time1)/(CLOCKS_PER_SEC*1.0);

		time1 = clock();		
		sampling_pushforward(rho, mu, xMap, yMap, n1, n2, totalMass);
		time2 = clock();
		time_pushforward += (time2 - time1)/(CLOCKS_PER_SEC*1.0);

		value = compute_w2(phi, dual, mu, nu, n1, n2);
		sigma = step_update(sigma, value, oldValue, gradSq, scaleUp, scaleDown, upper, lower);		
		oldValue = value;
		// sigma = fmax(sigma, .01);		
		values[i] = value;
		grad_norms[i] = gradSq;
		if(verbose){
			printf("iter %d, Stepsize: %e W2 value: %e h-1 residual: %e\n", i, sigma, value, gradSq);   
		}

	}
	for(i=0;i<n2;i++){
		for(j=0;j<n1;j++){
			x=(j+.5)/(n1*1.0);
			y=(i+.5)/(n2*1.0);
			phi[i*n1+j]=.5*(x*x+y*y) - phi[i*n1+j];
			dual[i*n1+j]=.5*(x*x+y*y) - dual[i*n1+j];
		}
	}
    if(verbose){
        printf("BaF algorithm detailed time: \n");
        printf("update potential: %.3f, convexify: %.3f, calc map: %.3f, pushforward: %.3f\n", time_update_potential, time_convexify, time_calc_map, time_pushforward);
    }
	return oldValue;
}

float fotGradient2d(struct fotSpace* otspace, float* grad, int n1, int n2, int verbose)
{
/*2d quadratic wasserstein distance using back-and-forth method with different normalization
INPUT: 
	n1, n2: 			dimension of the problem, note n1 is the direction for the fastest memory storage 

OUTPUT:
	grad:	gradient w.r.t. function f
	wd: 	quadratic wasserstein distance between f and g

TO DO:
	the Wasserstein distance is quite small for some data, which results gradient having extremly small
	amplitude. So we have to scale up the wasserstein distance a.w.a the gradient 
*/
	int i, pcount;
	pcount = n1 * n2;
	float wd, sigma, sum;
	float term = 0.;

	// normalize input signals
	sum = normalize((*otspace).mu, (*otspace).f, (*otspace).nu, (*otspace).g, pcount, 1);

	if(sum <= 0.)
		return 0;

	// init step size
	sigma = init_step_size((*otspace).mu, (*otspace).nu, pcount);
	// adjust step size with step scale
	sigma = (*otspace).step_scale / sigma;

	wd = compute_l2_fot2d((*otspace).mu, (*otspace).nu, (*otspace).phi, (*otspace).dual, (*otspace).rho, 
				(*otspace).xMap, (*otspace).yMap, 1, (*otspace).fftps, (*otspace).hull, sigma, (*otspace).nIter, 
				n1, n2, (*otspace).wd, (*otspace).gn, verbose);

	// gradient 
	for(i=0; i<pcount; i++){
		// if(normal_choice==1){// the shift normalize
		// 	term += 0.;
		// }else{
			term += (*otspace).mu[i] * (*otspace).dual[i] / pcount;
		// }
	}
	for(i=0; i<pcount; i++){
		// grad = ( h'(f)*phi )/ (int_ h(f)) - h'(f)/(int_ h(f))^2 *(int_ h(f)*phi)
		if(sum <= 0.)
			grad[i] *= 0.;
		else
			grad[i] *= ((*otspace).dual[i] - term)/sum;
	}

	return wd;
}
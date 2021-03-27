/*
* fot2d.h - header file for FOT
*
* Hao Zhang
* 01.01.2020
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stddef.h>
#include <string.h>
#include <fftw3.h>
#include "normalize.h"

#ifdef _OPENMP
#include <omp.h>
#endif

//extern "C"{
// #include <gsl/gsl_math.h>
// #include <gsl/gsl_interp2d.h>
// #include <gsl/gsl_spline2d.h>
//}

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------- */
/* declaration of FOT variale-structures */
/* and functions                         */
/* ------------------------------------- */

/* poisson solver */
struct poisson_solver{
	fftwf_plan dctIn;
	fftwf_plan dctOut;
	float *kernel;
	float *workspace;
};

float *create_negative_laplace_kernel2d(int n1, int n2);

struct poisson_solver create_poisson_solver_workspace2d(int n1, int n2);

void destroy_poisson_solver(struct poisson_solver fftps);

/* convex conjugate */
struct convex_hull{
	int *indices;
	int hullCount;
};

void alloc_hull(struct convex_hull *hull, int n);

void init_hull(struct convex_hull *hull, int n);

void destroy_hull(struct convex_hull *hull);

void add_point(float *u, struct convex_hull *hull, int i);

void get_convex_hull(float *u, struct convex_hull *hull, int n);

void compute_dual_indicies(int *dualIndicies, float *u, struct convex_hull *hull, int n);

void compute_dual(float *dual, float *u, int *dualIndicies, struct convex_hull *hull, int n);

void transpose_floats(float *transpose, float *data, int n1, int n2);

void compute_2d_dual(float *dual, float *u, struct convex_hull *hull, int n1, int n2);

void convexify(float *phi, float *dual, struct convex_hull *hull, int n1, int n2);

/* fast optimal transport kernels */
struct fotSpace
{
	float *xMap, *yMap;// domain
	float *f, *g; 		// original input signals
	float *mu, *nu;	// probability densities
	float *wd, *gn;// w2 values and H^-1 residuals (L2 norm of gradient) in iterations
	float *phi, *dual, *rho;
	struct poisson_solver fftps;
	struct convex_hull hull;
	float step_scale;
	int nIter;
};

void alloc_fotSpace_2d(struct fotSpace *fotspace, int n1, int n2);

void init_fotSpace_2d(struct fotSpace* fotspace, int n1, int n2, float* signal1, float* signal2);

void destroy_fotSpace_2d(struct fotSpace *fotspace);

float interpolate_function(float *function, float x, float y, int n1, int n2);

void calc_pushforward_map(float *xMap, float *yMap, float *dual, int n1, int n2);

void calc_pushforward_map_gsl(float *xMap, float *yMap, float *dual, int n1, int n2);

void sampling_pushforward(float *rho, float *mu, float *xMap, float *yMap, int n1, int n2, float totalMass);

float update_potential(struct poisson_solver fftps, float *phi, float *rho, float *nu, float sigma, int pcount);

float step_update(float sigma, float value, float oldValue, float gradSq, 
	float scaleUp, float scaleDown, float upper, float lower);

float compute_w2(float *phi, float *dual, float *mu, float *nu, int n1, int n2);

float compute_l2_fot2d(float *mu, float *nu, float *phi, float *dual, float *rho,  
		float *xMap, float *yMap, float totalMass, struct poisson_solver fftps, struct convex_hull hull, 
		float sigma, int maxIters, int n1, int n2, float *values, float *grad_norms, int verbose);

float fotGradient2d(struct fotSpace* otspace, float* grad, int n1, int n2, int verbose);
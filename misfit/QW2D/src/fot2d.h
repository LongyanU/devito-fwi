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
	fftw_plan dctIn;
	fftw_plan dctOut;
	double *kernel;
	double *workspace;
};

double *create_negative_laplace_kernel2d(int n1, int n2);

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

void add_point(double *u, struct convex_hull *hull, int i);

void get_convex_hull(double *u, struct convex_hull *hull, int n);

void compute_dual_indicies(int *dualIndicies, double *u, struct convex_hull *hull, int n);

void compute_dual(double *dual, double *u, int *dualIndicies, struct convex_hull *hull, int n);

void transpose_doubles(double *transpose, double *data, int n1, int n2);

void compute_2d_dual(double *dual, double *u, struct convex_hull *hull, int n1, int n2);

void convexify(double *phi, double *dual, struct convex_hull *hull, int n1, int n2);

/* fast optimal transport kernels */
struct fotSpace
{
	double *xMap, *yMap;// domain
	double *f, *g; 		// original input signals
	double *mu, *nu;	// probability densities
	double *wd, *gn;// w2 values and H^-1 residuals (L2 norm of gradient) in iterations
	double *phi, *dual, *rho;
	struct poisson_solver fftps;
	struct convex_hull hull;
	double step_scale;
	int nIter;
};

void alloc_fotSpace_2d(struct fotSpace *fotspace, int n1, int n2);

void init_fotSpace_2d(struct fotSpace* fotspace, int n1, int n2, double* signal1, double* signal2);

void destroy_fotSpace_2d(struct fotSpace *fotspace);

double interpolate_function(double *function, double x, double y, int n1, int n2);

void calc_pushforward_map(double *xMap, double *yMap, double *dual, int n1, int n2);

void calc_pushforward_map_gsl(double *xMap, double *yMap, double *dual, int n1, int n2);

void sampling_pushforward(double *rho, double *mu, double *xMap, double *yMap, int n1, int n2, double totalMass);

double update_potential(struct poisson_solver fftps, double *phi, double *rho, double *nu, double sigma, int pcount);

double step_update(double sigma, double value, double oldValue, double gradSq, 
	double scaleUp, double scaleDown, double upper, double lower);

double compute_w2(double *phi, double *dual, double *mu, double *nu, int n1, int n2);

double compute_l2_fot2d(double *mu, double *nu, double *phi, double *dual, double *rho,  
		double *xMap, double *yMap, double totalMass, struct poisson_solver fftps, struct convex_hull hull, 
		double sigma, int maxIters, int n1, int n2, double *values, double *grad_norms, int verbose);

double fotGradient2d(struct fotSpace* otspace, double* grad, int n1, int n2, int verbose);
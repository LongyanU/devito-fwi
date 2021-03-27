#ifndef NORMALIZE_H
#define NORMALIZE_H

#ifndef EPS
#define EPS 1e-6
#endif

double normalize(double *fn, double *f, double *gn, double *g, int n, int flag);

#endif
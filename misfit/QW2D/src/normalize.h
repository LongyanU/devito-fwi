#ifndef NORMALIZE_H
#define NORMALIZE_H

#ifndef EPS
#define EPS 1e-6
#endif

float normalize(float *fn, float *f, float *gn, float *g, int n, int flag);

#endif
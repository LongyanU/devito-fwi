
CC=gcc
MPICC=mpicc 
CFLAGS = -Wall -O3 -std=c11
LDFLAGS = -lm -lfftw3f -fopenmp

SOURCES = fot2d.c normalize.c w2.c

bfm2d: 
	$(CC) $(SOURCES) -o $@  $(CFLAGS) $(LDFLAGS)

clean: 
	rm -f bfm2d
	rm -f $(OBJECTS)	
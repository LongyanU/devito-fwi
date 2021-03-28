
#include "fot2d.h"

#define MAX_STRLEN 1000

int main(int argc, char **argv){

	int n1, n2, niter;
	int write_adj;
	float step_scale;

	n1 = atoi(argv[1]);
	n2 = atoi(argv[2]);
	niter = atoi(argv[3]);
	step_scale = atof(argv[4]);
	write_adj = atoi(argv[5]);
	char *dir = argv[6];	

	// Local variables
	char str[MAX_STRLEN];
	float *obs = malloc(n1 * n2 * sizeof(float));
	float *syn = malloc(n1 * n2 * sizeof(float));
	float *adj = malloc(n1 * n2 * sizeof(float));
	memset(adj, 1, n1 * n2 * sizeof(float));
	FILE *fp;

	// Create local BFM solver
	struct fotSpace otspace = {0};
	otspace.nIter = niter;
	otspace.step_scale = step_scale;
	alloc_fotSpace_2d(&otspace, n1, n2);
	float w = 0;

	/* Read obs and syn data (binary) */

	sprintf(str, "%s/obs_data", dir);
	fp = fopen(str, "rb");
	if(fread(obs, sizeof(float), n1*n2, fp) != (size_t)(n1*n2)){
		fprintf(stderr, "Failed to reading data from %s\n", str);
		exit(EXIT_FAILURE);
	}
	fclose(fp);
	sprintf(str, "%s/syn_data", dir);
	fp = fopen(str, "rb");
	if(fread(syn, sizeof(float), n1*n2, fp) != (size_t)(n1*n2)){
		fprintf(stderr, "Failed to reading data from %s\n", str);
		exit(EXIT_FAILURE);
	}
	fclose(fp);

	init_fotSpace_2d(&otspace, n1, n2, syn, obs);
	w = fotGradient2d(&otspace, adj, n1, n2, 1);

	if(write_adj){
		sprintf(str, "%s/grad_data", dir);
		fp = fopen(str, "wb");
		if(fwrite(adj, sizeof(float), n1*n2, fp) != (size_t)(n1*n2)){
			fprintf(stderr, "Failed to write data to %s\n", str);
			exit(EXIT_FAILURE);
		}
		fclose(fp);
	}

	sprintf(str, "%s/loss", dir);
	fp = fopen(str, "w");
	fprintf(fp, "%e\n", w);
	fclose(fp);

	free(syn);
	free(obs);
	free(adj);
	destroy_fotSpace_2d(&otspace);
	return 0;
}
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <immintrin.h>

#include "pagerank.h"
#define DEBUG 0

static ssize_t g_nthreads = 1;
size_t g_chunk;
size_t g_chunk_elements;
size_t npages;
size_t nelements;
double inv_damp;
double inv_n;

typedef struct {
	size_t id;
	double* result;
	const double* a;
	const double* b;
	const size_t extra;
} worker_args;

typedef struct {
	size_t id;
	double* result;
	double* M;
	double dampener;
} worker_dampen_args;

double* matrix_mul(double* matrix_a, double* vector, size_t npages, double* result);
void dampen(double* Mhat, double *M, double dampener);

void pagerank(node* list, size_t pages, size_t nedges, size_t nthreads, double dampener) {

	npages = pages;

	// set limit on number of threads
	if (nthreads > npages) {
		nthreads = npages;
	}
	g_nthreads = nthreads;

	nelements = npages * npages;
	g_chunk = npages/g_nthreads;
	g_chunk_elements = nelements/g_nthreads;

	// set constants for calculation
	inv_n = 1.0/npages;
	inv_damp = (1.0-dampener)/npages;

	// TODO: Calculate IN sets
	// int nIn[npages];
	// node **in = (node**) malloc(nelements * sizeof(node*));


	// Construct Probability matrix (M)
	//probMatrix(M, list, npages);

	char **names = (char**) malloc(npages * sizeof(char*));
	double *M = (double*) malloc(nelements * sizeof(double));
	node *current = list;
	for (int i = 0; i < npages; i++) {
		node* current_j = list;
		for (int j = 0; j < npages; j++) {
			// if OUT[j] is 0;
			// DEBUG printf("OUT[%d] = %zu\n", i, current->page->noutlinks);
			if (current_j->page->noutlinks == 0) {
				// DEBUG printf("OUT[%d] = 0: [%d][%d]\n", j, i, j);
				M[i * npages + j] = inv_n;
			}
			// Use the IN sets to see if there are any links
			else {
				bool link = false;
				// iterate through in list, and try and see if j is
				//  in i's in-list
				node* current_in = current->page->inlinks;
				while (true) {
					// set nIn[i] to counter and break if end of LL is reached.
					if (!current_in) {
						// printf("----NULL----\n");
						break;
					}
					if (current_in->page->name == current_j->page->name) {
						link = true;
						break;
					}
					current_in = current_in->next;
				}

				if (link == true) {
					// if j links to i
					// DEBUG printf("found a link: [%d][%d]\n", i, j);
					M[i * npages + j] = 1.0/current_j->page->noutlinks;
				}
				else {
					// DEBUG printf("No links: [%d][%d]\n", i, j);
					// No links to page
					M[i * npages + j] = 0;
				}
			}
			printf("[%d][%d]: %lf\n", i, j, M[i * npages + j]);
			current_j = current_j->next;
		}
		current = current->next;
	}

	// print M
	if(DEBUG) {
		for (int i = 0; i < npages; i++) {
			printf("%s ", names[i]);
		}
		printf("\n----------\n");
		for (int i = 0; i < npages; i++) {
			for (int j = 0; j < npages; j++) {
				printf("%lf ", M[j * npages + i]);
			}
			printf("\n");
		}
		printf("-------- \n");
	}

	// TODO: Construct pagerank score matrix and initialise to 1/N
	double P[npages];
	for (int i = 0; i < npages; i++) {
		P[i] = inv_n;
	}


	// ------------ Calculate Mhat ----------------------------------------
	//---------------------------------------------------------------------

	double *Mhat = (double*) malloc(nelements * sizeof(double));
	if(npages < 5) {
		for (int i = 0; i < nelements; i++) {
			Mhat[i] = (M[i] * dampener) + inv_damp;
		}
	} else {
		dampen(Mhat, M, dampener);
	}


	// print M(hat)
	if(DEBUG) {
		for (int i = 0; i < npages; i++) {
			printf("%s ", names[i]);
		}
		printf("\n-----Mhat-----\n");
		for (int i = 0; i < npages; i++) {
			for (int j = 0; j < npages; j++) {
				printf("%lf ", Mhat[i * npages + j]);
			}
			printf("\n");
		}
		printf("-------- \n");
	}

	// ---------------- Main iteration step: ------------------------------
	//---------------------------------------------------------------------

	int count = 0;
	double newP[npages];

	while(true) {
		count ++;
		double vector_norm_sum = 0.0;

		if(npages > 5) {
			matrix_mul(Mhat, P, npages, newP);
			for (int i = 0; i < npages; i++) {
				double diff = (newP[i] - P[i]);
				vector_norm_sum += diff*diff;
			}
			memcpy(P, newP, sizeof(P));
		}

		else {
			for (int i = 0; i < npages; i++) {
				newP[i] = 0;
				// iterate through ith row of M and multiply with P[j]
				for (int j = 0; j < npages; j++) {
					if(DEBUG) {
						printf("%lf * %lf + \n", P[j], Mhat[i * npages + j]);
					}
					newP[i] += P[j] * Mhat[i * npages + j];
				}
				double diff = (newP[i] - P[i]);
				vector_norm_sum += diff*diff;
			}
			memcpy(P, newP, sizeof(P));
		}

		// DEBUG print results of each iteration
		if(DEBUG) {
			printf("-------\niteration %d: vector normal = %lf\n", count, sqrt(vector_norm_sum));
			current = list;
			for (size_t i = 0; i < npages; i++) {
				printf("%s %.8lf\n", current->page->name, P[i]);
				current = current->next;
			}
		}

		if (sqrt(vector_norm_sum) < EPSILON) {
			break;
		}

		//printf("%d\n", count);
	}

	// Print scores for each page
	current = list;
	for (size_t i = 0; i < npages; i++) {
		printf("%s %.8lf\n", current->page->name, P[i]);
		current = current->next;
	}

	free(names);
	free(M);
	free(Mhat);

}

//---------------------------------------------------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------


/**
 * Returns the product of the column vector and matrix provided
 */
void* worker_mul(void* args) {
	worker_args* wargs = (worker_args*) args;

	size_t id = wargs->id;
	size_t npages = wargs->extra;
	const size_t start = id * g_chunk;
	const size_t end = id == g_nthreads - 1 ? npages : (id + 1) * g_chunk;
	const double* Mhat = wargs->a;
	const double* P = wargs->b;

	double* result = wargs->result;

	for (int i = start; i < end; i++) {
		result[i] = 0;
		// iterate through ith row of M and multiply with P[j]
		for (int j = 0; j < npages; j++) {
			if(DEBUG) {
				printf("%lf * %lf + \n", P[j], Mhat[i * npages + j]);
			}
			result[i] += P[j] * Mhat[i * npages + j];
		}
	}
	return NULL;
}


double* matrix_mul(double* matrix_a, double* vector, size_t npages, double* result) {

	worker_args args[g_nthreads];
	pthread_t thread_ids[g_nthreads];

	for (size_t i = 0; i < g_nthreads; i++) {
		args[i] = (worker_args) {
			.a = matrix_a,
			.b = vector,
			.id = i,
			.result = result,
			.extra = npages,
		};
	}

	// Launch threads
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_create(thread_ids + i, NULL, worker_mul, args + i);
	}

	// Wait for threads to finish
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_join(thread_ids[i], NULL);
	}

	return result;
}


void* worker_dampen(void* args) {
	worker_dampen_args* wargs = (worker_dampen_args*) args;

	size_t id = wargs->id;
	const size_t start = id * g_chunk_elements;
	const size_t end = id == g_nthreads - 1 ? nelements : (id + 1) * g_chunk_elements;
	double* Mhat = wargs->result;
	const double* M = wargs-> M;
	const double dampener = wargs->dampener;

	for (int i = start; i < end; i++) {
		Mhat[i] = (M[i] * dampener) + inv_damp;
	}
	return NULL;
}

void dampen(double* Mhat, double* M, double dampener) {
	worker_dampen_args args[g_nthreads];
	pthread_t thread_ids[g_nthreads];

	for (size_t i = 0; i < g_nthreads; i++) {
		args[i] = (worker_dampen_args) {
			.id = i,
			.result = Mhat,
			.dampener = dampener,
			.M = M,
		};
	}

	// Launch threads
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_create(thread_ids + i, NULL, worker_dampen, args + i);
	}

	// Wait for threads to finish
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_join(thread_ids[i], NULL);
	}
}

/*
######################################
### DO NOT MODIFY BELOW THIS POINT ###
######################################
*/

int main(int argc, char** argv) {

	/*
	######################################################
	### DO NOT MODIFY THE MAIN FUNCTION OR HEADER FILE ###
	######################################################
	*/

	config conf;

	init(&conf, argc, argv);

	node* list = conf.list;
	size_t npages = conf.npages;
	size_t nedges = conf.nedges;
	size_t nthreads = conf.nthreads;
	double dampener = conf.dampener;

	pagerank(list, npages, nedges, nthreads, dampener);

	release(list);

	return 0;
}

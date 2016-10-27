#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <immintrin.h>

#include "pagerank.h"
#define DEBUG 0

static ssize_t g_nthreads = 1;
size_t g_chunk = 0;
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
	node *list;
	const size_t npages;
} worker_probmatrix_args;

double* matrix_mul(double* matrix_a, double* vector, size_t npages, double* result);
void probMatrix(double *M, node *list, int npages);

void pagerank(node* list, size_t npages, size_t nedges, size_t nthreads, double dampener) {

	// set limit on number of threads
	if (nthreads > npages) {
		nthreads = npages;
	}

	g_nthreads = nthreads;
	g_chunk = npages/g_nthreads;

	// set constants for calculation
	inv_n = 1.0/npages;
	inv_damp = (1.0-dampener)/npages;
	int nelements = npages * npages;

	// TODO: Calculate IN sets
	// int nIn[npages];
	// node **in = (node**) malloc(nelements * sizeof(node*));

	char **names = (char**) malloc(npages * sizeof(char*));
	node *current;

	// Construct Probability matrix (M)
	double *M = (double*) malloc(nelements * sizeof(double));
	// probMatrix(M, list, npages);

	for (int i = 0; i < npages; i++) {
		node* current_j = list;
		for (int j = 0; j < npages; j++) {
			// if OUT[j] is 0;
			// DEBUG printf("OUT[%d] = %zu\n", i, current->page->noutlinks);
			if (current_j->page->noutlinks == 0) {
				// DEBUG printf("OUT[%d] = 0: [%d][%d]\n", j, i, j);
				M[j * npages + i] = inv_n;
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
					M[j * npages + i] = 1.0/current_j->page->noutlinks;
				}
				else {
					// DEBUG printf("No links: [%d][%d]\n", i, j);
					// No links to page
					M[j * npages + i] = 0;
				}
			}
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


	/* TODO: Construct M(hat) by:
		- For each value of M (we don't need the original again):
			- multiply by M
			- add (1-d)/N
	*/
	double *Mhat = (double*) malloc(nelements * sizeof(double));
	for (int i = 0; i < nelements; i++) {
		Mhat[i] = (M[i] * dampener) + inv_damp;
	}

	// print M(hat)
	if(DEBUG) {
		for (int i = 0; i < npages; i++) {
			printf("%s ", names[i]);
		}
		printf("\n-----Mhat-----\n");
		for (int i = 0; i < npages; i++) {
			for (int j = 0; j < npages; j++) {
				printf("%lf ", Mhat[j * npages + i]);
			}
			printf("\n");
		}
		printf("-------- \n");
	}

	/* TODO: Main iteration step:
	While(true):
		- set sum to 0
		- multiply P and M(hat)
			- while in each cell, save old value temporarily
			- calculate the square of the difference between the new and old values
			- add to a sum variable
		- If sqrt(sum) is less than epsilon, end the loop
	*/

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
						printf("%lf * %lf + \n", P[j], Mhat[j * npages + i]);
					}
					newP[i] += P[j] * Mhat[j * npages + i];
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

/**
 * Returns new matrix that is the result of
 * multiplying the two matrices together.
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
				printf("%lf * %lf + \n", P[j], Mhat[j * npages + i]);
			}
			result[i] += P[j] * Mhat[j * npages + i];
		}
	}
	return NULL;
}


/*
######################################
### DO NOT MODIFY BELOW THIS POINT ###
######################################
*/

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

void* worker_probMatrix(void* args) {
	worker_probmatrix_args* wargs = (worker_probmatrix_args*) args;

	size_t id = wargs->id;
	node *list = wargs->list;
	const int npages = wargs->npages;
	double* M = wargs->result;

	const size_t start = id * g_chunk;
	const size_t end = id == g_nthreads - 1 ? npages : (id + 1) * g_chunk;

	node* current = list;
	if(npages > 40) {
		printf("thread %zu doing rows %zu -> %zu", id, start, end);
	}
	for (int i = start; i < end; i++) {
		node* current_j = list;
		for (int j = 0; j < npages; j++) {
			// DEBUG printf("OUT[%d] = %zu\n", i, current->page->noutlinks);
			if (current_j->page->noutlinks == 0) {
				// DEBUG printf("OUT[%d] = 0: [%d][%d]\n", j, i, j);
				M[j * npages + i] = inv_n;
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
					M[j * npages + i] = 1.0/current_j->page->noutlinks;
				}
				else {
					// DEBUG printf("No links: [%d][%d]\n", i, j);
					// No links to page
					M[j * npages + i] = 0;
				}
			}
			current_j = current_j->next;
		}
		current = current->next;
	}

	return NULL;

}

void probMatrix(double *M, node *list, int npages) {
	worker_probmatrix_args args[g_nthreads];
	pthread_t thread_ids[g_nthreads];

	for (size_t i = 0; i < g_nthreads; i++) {
		args[i] = (worker_probmatrix_args) {
			.id = i,
			.result = M,
			.list = list,
			.npages = npages,
		};
	}

	// Launch threads
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_create(thread_ids + i, NULL, worker_probMatrix, args + i);
	}

	// Wait for threads to finish
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_join(thread_ids[i], NULL);
	}
}


//------------------------------------------------------------------------
//------------------------------------------------------------------------
//------------------------------------------------------------------------
//------------------------------------------------------------------------

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

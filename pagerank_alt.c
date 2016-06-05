#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <immintrin.h>

#include "pagerank.h"
#define DEBUG 0

void pagerank(node* list, size_t npages, size_t nedges, size_t nthreads, double dampener) {

	// set limit on number of threads
	if (nthreads > npages) {
		nthreads = npages;
	}

	// set constants for calculation
	double inv_n = 1.0/npages;
	double inv_damp = (1.0-dampener)/npages;

	// required values:
	// - list of webpages (node* list)
	// - number of pages (npages)
	// - Vector of pagerank scores	-TODO
	// - dampening factor (dampener)
	// - convergence threshold (EPSILON)
	// - IN(p): set of all pages in S which link to page P
	// - OUT(P): set of all pages which P links to
	// - M: matrix of transition probabilities
	// - M(hat): matrix for multiplication with P in each iteration

	int nelements = npages * npages;

	// TODO: Calculate IN sets
	int nIn[npages];
	node* in[npages][npages];

	char *names[npages];
	node *current;

	// iterate through inlinks until a null item is reached:
	current = list;
	for (int i = 0; i < npages; i++) {
		names[i] = current->page->name;
		// DEBUG printf("currently finding inlinks for %s\n", current->page->name);

		node* current_in = current->page->inlinks;
		int count = 0;
		while (true) {
			// set nIn[i] to counter and break if end of LL is reached.
			if (!current_in) {
				// printf("----NULL----\n");
				nIn[i] = count;
				break;
			}
			// printf("new inlink! %s <- %s\n", current->page->name, current_in->page->name);
			// add pointer to current element to 2d array
			in[i][count] = current_in;
			// increment for next loop
			count++;
			current_in = current_in->next;
		}
		current = current->next;
	}

	// print in list
	if(DEBUG) {
		for (int i = 0; i < npages; i++) {
			printf("%s ", names[i]);
		}
		printf("\n----------\n");
		for (int i = 0; i < npages; i++) {
			for (int j = 0; j < nIn[i]; j++) {
				printf("%s ", in[i][j]->page->name);
			}
			printf("\n");
		}
	}

	// Construct Probability matrix (M)
	double M[nelements];
	current = list;
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
				for (int k = 0; k < nIn[i]; k++) {
					// iterate through in list, and try and see if j is
					//  in i's in-list
					if (in[i][k]->page->name == current_j->page->name) {
						link = true;
						break;
					}
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
	double Mhat[nelements];
	for (int i = 0; i < nelements; i++) {
		Mhat[i] = (M[i] * dampener) + inv_damp;
	}

	// print M(hat)
	if(DEBUG) {
		for (int i = 0; i < npages; i++) {
			printf("%s ", names[i]);
		}
		printf("\n----------\n");
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
		for (int i = 0; i < npages; i++) {
			newP[i] = 0;
			// iterate through ith row of M and multiply with P[j]
			for (int j = 0; j < npages; j++) {
				// printf("%lf * %lf + \n", P[j], Mhat[j * npages + i]);
				newP[i] += P[j] * Mhat[j * npages + i];
			}
			double diff = (newP[i] - P[i]);
			vector_norm_sum += diff*diff;
		}

		memcpy(P, newP, sizeof(P));


		// DEBUG print results of each iteration
		if(DEBUG) {
			printf("-------\niteration %d: vector normal = %lf\n", count, sqrt(vector_norm_sum));
			current = list;
			for (size_t i = 0; i < npages; i++) {
				printf("%s %.8lf\n", current->page->name, P[i]);
				current = current->next;
			}
		}
		// END DEBUG

		if (sqrt(vector_norm_sum) < EPSILON) {
			break;
		}
	}

	// Print scores for each page
	current = list;
	for (size_t i = 0; i < npages; i++) {
		printf("%s %.8lf\n", current->page->name, P[i]);
		current = current->next;
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

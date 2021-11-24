#define VARIABLES  7
#define SAMPLES    9

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "correlation.h"
#include <math.h>

float *rand_vector(int);
void print_matrix(float *, int, int);

int main() {
    int variables=VARIABLES;
    int samples=SAMPLES;
    float *A = rand_vector(variables*samples);
    float *B = gpu_correlation((float *) A, variables,samples);
    print_matrix(A, variables, samples);
    print_matrix(B, variables, variables);
    free(A);
    free(B);
}

// just to visualize a matrix.
void print_matrix(float *A, int m, int n) {
    for(int row=0; row<m; row++) {
        for(int col=0; col<n; col++)
            printf("%- 13.4e", A[row*n + col]);
        printf("\n");
    }
    printf("%dx%d Matrix\n\n", m, n);
}

float *rand_vector(int n) {
    float *out = (float *) malloc(n * sizeof(float));
    srand(time(0));
    for (int i=0; i<n; i++)
        out[i] = (float) rand() / RAND_MAX * 2 - 1;
    return out;
}

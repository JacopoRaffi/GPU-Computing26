#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <time.h>

#include "include/my_time_lib.h"

#define dtype double
#define WARMUP 2
#define NITER 10

// Function to fill a pre-allocated matrix with random values
void fill_matrix(int rows, int cols, dtype *matrix) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*cols +j] = (float)(rand() % 100) / 10.0;
        }
    }
}

void init_matrix(int rows, int cols, dtype *matrix, dtype val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*cols +j] = val;
        }
    }
}

void my_gemm(int n, int k, int m, dtype *A, dtype *B, dtype *C) {

    for (int i=0; i<n; i++)
	for (int j=0; j<m; j++)
	    for (int h=0; h<k; h++)
		C[i*m +j] += A[i*k +h] * B[h*m +j];
}


int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Usage: %s n\n\n"
               "Where:\n"
               "\tmatrix A has size n x n\n"
               "\tmatrix B has size n x n\n"
               "\tresult matrix C has size n x n\n",
               argv[0]);
        return 1;
    }
    // Generate random matrices
    dtype *A, *B, *C;
    double timers[NITER];

    /* Generate now two square matrices A and B of size (2^n) and fill them with random doubles.
     *  After this, compute the matrix multiplication C = A x B using OpenBLAS: cblas_dgemm()
     *
     * Tasks:
     *      1) Record CPU time on time[] vector. 
     *      2) Compute their arithmetic mean and geometric mean.
     *
     * NOTE:
     *      1) Fill A and B with random doubles
     *      2) Init C with 0.0
     */
    int n = atoi(argv[1]);
    int k = n;
    int m = n;
    fprintf(stdout, "Input sizes are %d x %d x %d\n", n, k, m);

    A = (dtype*)malloc(sizeof(dtype)*n*k);
    B = (dtype*)malloc(sizeof(dtype)*k*m);
    C = (dtype*)malloc(sizeof(dtype)*n*m);

    srand(time(NULL));
    fill_matrix(n, k, A);
    fill_matrix(k, m, B);
    init_matrix(n, m, C, 0.0);


    double iter_time, a_mean, g_mean;

    TIMER_DEF(0);
    
    for (int i=-WARMUP; i<NITER; i++) {

        // Perform C = A * B using BLAS
        TIMER_START(0);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.0, A, k, B, n, 0.0, C, n);
        TIMER_STOP(0);

        iter_time = TIMER_ELAPSED(0) / 1.e6;
        if( i >= 0) timers[i] = iter_time;

        printf("Iteration %d tooks %lfs\n", i, iter_time);
        init_matrix(n, m, C, 0.0);
    }


    /*  EX1: Here we compute the vectors' arithmetic mean and geometric mean; 
     *  these functions must be implemented inside
     *   of the library "src/my_time_lib.c" (and their headers in "include/my_time_lib.h").
     */
    a_mean = arithmetic_mean(timers, NITER);
    g_mean = geometric_mean(timers, NITER);
    printf(" %10s | %10s | %10s |\n", "v name", "arithmetic mean", "geometric mean");
    printf(" %10s | %10f | %10f |\n", "time", a_mean, g_mean);


    /*  EX2: Here we compute the FLOPS using arithmetic mean of time vector
     *
     */
    int nflop = n * k * m;;
    double flops;
    fprintf(stdout, "\nEach gemm required n*k*m = %d floating point operations.\n", nflop);
    flops = nflop / a_mean;
    fprintf(stdout, "The OpenBLAS gemm achieved %lf MFLOP/s\n", flops / 1.e6);
  


    /*  EX2: Implement your own general matrix multiplication
     *  Compare your performance with the OpenBLAS one
     *  Note: Perform a correctness check   
     */
    fprintf(stdout, "\nMy GEMM implementation:\n");
    for (int i=-WARMUP; i<NITER; i++) {

        // Perform C = A * B using BLAS
        TIMER_START(0);
	    my_gemm(n, k, m, A, B, C);        
        TIMER_STOP(0);

        double iter_time = TIMER_ELAPSED(0) / 1.e6;
        if( i >= 0) timers[i] = iter_time;

        printf("Iteration %d tooks %lfs\n", i, iter_time);
        init_matrix(n, m, C, 0.0);
    }

    a_mean = arithmetic_mean(timers, NITER);
    fprintf(stdout, "Arithmetic Mean: %lf\n", a_mean);

    flops = nflop / a_mean;
    fprintf(stdout, "My GEMM implementation achieved %lf MFLOP/s\n", flops / 1.e6);

    return(0);
}


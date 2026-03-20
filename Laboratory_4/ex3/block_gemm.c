#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <time.h>

#include "include/my_time_lib.h"

#define dtype double
#define WARMUP 2
#define NITER 10



int main(int argc, char *argv[]) {

    if (argc < 1) {
        printf("Usage: %s n blk_n\n\nWhere:\tmatrix A has size nxn\n\tmatrix B has size nxn\n\tresult matrix C has size nxn\n\tblock_n is the size of the block multiplication\n", argv[0]);
        return(1);
    }

    // Generate random matrices
    dtype *A, *B, *C;
    double timers[NITER];
    
    /* EX3: Generate now two square matrices A and B of size (2^n) and fill them with random doubles.
     *  After this, 
     * 1) implement GEMM by splitting the original matrix into blocks
     * 2) record the time
     * 3) Measure FLOP/s
     * NOTE:
     *      1) Fill A and B with random doubles
     *      2) Init C with 0.0
     *      3) Perform a correctness check
     */

    /* |========================================| */
    /* |           Put here your code           | */
    /* |========================================| */


    return(0);
}


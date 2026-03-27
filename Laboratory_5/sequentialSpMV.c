#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WARMUP 2
#define NITER 10

#include "include/my_time_lib.h"



int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <rows> <cols> <nnz>\n", argv[0]);
        return 1;
    }

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    int nnz = atoi(argv[3]);
    double timers[NITER];


    /* EX1: Write a sequential SpMV algorithm that multiply a randomly generated COO with a dense vector were all the entrances are set to 1.
     * Three steps: 
     * a) allocate and generate COO
     * b) generate a vector with 1
     * c) implement spmv with COO format 
    */

    /* |========================================| */
    /* |           Put here your code           | */
    /* |========================================| */


    /* EX1: Do the same by using a CSR data-format
     * Two steps: 
     * a) allocate and generate CSR: COO to CSR
     * b) implement spmv with CSR format 
    */

    /* |========================================| */
    /* |           Put here your code           | */
    /* |========================================| */

    /* EX1: Check if spmv_COO and spmv_CSR have the same output
    */

    /* |========================================| */
    /* |           Put here your code           | */
    /* |========================================| */


  
    /* EX1: Banchmark the runtime of spmv_COO and spmv_CSR separately. With 2 warmups and 10 iterations, calculate their arithmetic mean
    */

    /* |========================================| */
    /* |           Put here your code           | */
    /* |========================================| */

    /* EX1: Banchmark the effective bandwidths of spmv_COO and spmv_CSR separately.
    */

    /* |========================================| */
    /* |           Put here your code           | */
    /* |========================================| */



    /* Note, free the memory */

    /* |========================================| */
    /* |           Put here your code           | */
    /* |========================================| */

    return 0;
}


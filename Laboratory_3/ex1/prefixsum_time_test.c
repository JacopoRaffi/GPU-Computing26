#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define dtype int
#define NITER 10
#define WARMUP 2

#include "include/my_time_lib.h"

int main(int argc, char *argv[]) {

    // Example $./prefixsum 32 
    if (argc != 2) {
        printf("Usage: %s n \n\n"
               "Where:\n"
               "\tarray A has size n\n"
               "\tresult array P has size n\n",
               argv[0]);
        return 1;
    }

    dtype *A, *P;
    double timers[NITER];

    /* Ex1: Generate now one array A and fill it with random ints.
     *  After this, compute the prefix sum P 
     *
     * Tasks:
     *      1) Record CPU time on time[] vector. 
     *      2) Compute their arithmetic mean and geometric mean.
     *
     * NOTE:
     *      1) Fill A with random int
     *      2) You could print input and result for small arrays to check the correctness
     */

    /* |========================================| */
    /* |           Put here your code           | */
    /* |========================================| */



    /* Ex1:  Here we compute the vectors' arithmetic mean and geometric mean; 
     *  these functions must be implemented inside
     *   of the library "src/my_time_lib.c" (and their headers in "include/my_time_lib.h").
     */
    double a_mean = 0.0, g_mean = 0.0;


    /* |========================================| */
    /* |           Put here your code           | */
    /* |========================================| */


    printf(" %10s | %10s | %10s |\n", "v name", "arithmetic mean", "geometric mean");
    printf(" %10s | %10f | %10f |\n", "time", a_mean, g_mean);

    return(0);
}

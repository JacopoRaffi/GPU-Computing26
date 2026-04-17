#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>

#define NITER 10
#define WARMUP 4

#define NPROBS 3

#define TIMER_DEF     struct timeval temp_1, temp_2

#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)

#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)

#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)*1000.0 +(temp_2.tv_usec-temp_1.tv_usec)/1000.0) // return ms

#define DBG_CHECK { printf("DBG_CHECK: file %s at line %d\n", __FILE__, __LINE__ ); }
// #define DEBUG
// #define BLK_DISPACH

#define RUN_SOLUTIONS

#define BLK_SIZE 32
#define GRD_SIZE 16

#define STR(s) #s
#define XSTR(s) STR(s)
#define dtype float

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("  Memory Clock rate:           %.0f Mhz\n", devProp.memoryClockRate * 1e-3f);

    printf("  Memory Bus Width:            %d bit\n",devProp.memoryBusWidth);

    printf("  Peak Memory Bandwidth:       %7.3f GB/s\n",2.0*devProp.memoryClockRate*(devProp.memoryBusWidth/8)/1.0e6);

    printf("  Multiprocessors:             %3d\n",devProp.multiProcessorCount);
    printf("  Maximum number of threads per multiprocessor:  %d\n",devProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",devProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           devProp.maxThreadsDim[0], devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           devProp.maxGridSize[0], devProp.maxGridSize[1],devProp.maxGridSize[2]);
    printf("  Total amount of shared memory per block:       %zu bytes\n", devProp.sharedMemPerBlock);
    return;
}

float arithmetic_mean(float *v, int len) {

    float mu = 0.0;
    for (int i=0; i<len; i++)
        mu += (float)v[i];
    mu /= (float)len;

    return(mu);
}

float compute_std(float *arr, int n, float mean) {
    float variance = 0.0;

    for (int i = 0; i < n; i++) {
        float diff = arr[i] - mean;
        variance += diff * diff;
    }

    variance /= (n - 1);
    return sqrt(variance);
}


__device__ uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

__global__
void example_kernel(int n, dtype *a, dtype* b, dtype* c)
{
  if (threadIdx.x==0)
      printf("block %d runs on sm %d\n", blockIdx.x, get_smid());

  // [ ... ]
}



__global__
void Problem_solution_layout1(int n, dtype *a, dtype* b, dtype* c)
{
#ifdef BLK_DISPACH
  if (threadIdx.x==0)
      printf("block %d runs on sm %d\n", blockIdx.x, get_smid());
#endif

  int tid = blockIdx.x*blockDim.x + threadIdx.x;
//   printf("%d = tid = blockIdx.x*blockDim.x + threadIdx.x = %d * %d + %d\n", tid, blockIdx.x, blockDim.x, threadIdx.x);
  int nthreads = gridDim.x*blockDim.x;
  int tsize = ((n % nthreads) == 0) ? (n/nthreads) : (n/nthreads)+1 ;
  int accessindex;

  for (int i=0; i<tsize; i++) {
    accessindex = tid + i*nthreads; //Each thread processes a contiguous memory access, rather than a chunk
    if (accessindex < n)
        c[accessindex] = a[accessindex] + b[accessindex]; // (dtype)tid;
  }
}

__global__
void Problem_solution_layout2(int n, dtype *a, dtype* b, dtype* c, int h)
{
#ifdef BLK_DISPACH
  if (threadIdx.x==0)
      printf("block %d runs on sm %d\n", blockIdx.x, get_smid());
#endif  
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;

    int tsize = ((n % nthreads) == 0) ? (n/nthreads) : (n/nthreads)+1 ;
    int accessindex;

    for (int i = 0; i < tsize; i++) {
        accessindex = tid + i * nthreads;

        if (accessindex < n) {
            int shifted = accessindex + h;
            if (shifted >= n) shifted -= n;   // wrap-around safely

            c[accessindex] = a[shifted] + b[shifted];
        }
    }
}

__global__
void Problem_solution_layout3(int n, dtype *a, dtype* b, dtype* c)
{
#ifdef BLK_DISPACH
  if (threadIdx.x==0)
      printf("block %d runs on sm %d\n", blockIdx.x, get_smid());
#endif

  int tid = blockIdx.x*blockDim.x + threadIdx.x;
//   printf("%d = tid = blockIdx.x*blockDim.x + threadIdx.x = %d * %d + %d\n", tid, blockIdx.x, blockDim.x, threadIdx.x);
  int nthreads = gridDim.x*blockDim.x;
  int tsize = ((n % nthreads) == 0) ? (n/nthreads) : (n/nthreads)+1 ;
  int accessindex;

  for (int i=0; i<tsize; i++) {
    accessindex = tid*tsize + i;  //Each thread processes a contiguous chunk
    if (accessindex < n)
        c[accessindex] = a[accessindex] + b[accessindex]; // (dtype)tid;
  }
}


int main(int argc, char *argv[]) {

  printf("======================================= Device properties ========================================\n");

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  int dev, SM_num;
  for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printDevProp(deviceProp);
    SM_num = deviceProp.multiProcessorCount;

  }


  printf("====================================== Problem computations ======================================\n");
  // =========================================== Set-up the problem ============================================

  if (argc < 2) {
    printf("Usage: lab8_ex5 n\n");
    return(1);
  }
  printf("argv[1] = %s\n", argv[1]);

  // ---------------- set-up the problem size -------------------

  for (int n = atoi(argv[1]); n <= 22; n += 2){   // vary the array size, starting from 2^12, to 2^22
    int len = (1<<n), i;
    printf("===================================== Ex6: Find best BLK and GRD dimensions ====================================\n");
    printf("n = %d --> len = 2^(n) = %d\n", n, len);
    printf("dtype = %s\n", XSTR(dtype));

    // ------------------ set-up the variables ---------------------

    TIMER_DEF;
    float error, cputime, gputime_iter; 
    float gputime_layout1[NITER], gputime_layout2[NITER], gputime_layout3[NITER];
  

    // ------------------- set-up the problem -------------------

    dtype *a, *b, *CPU_c, *GPU_c;
    a = (dtype*)malloc(sizeof(dtype)*len);
    b = (dtype*)malloc(sizeof(dtype)*len);
    CPU_c = (dtype*)malloc(sizeof(dtype)*len);
    GPU_c = (dtype*)malloc(sizeof(dtype)*len);
    time_t t;
    srand((unsigned) time(&t));

    int typ = (strcmp( XSTR(dtype) ,"int")==0);
    if (typ) {
        // here we generate random ints
        int rand_range = (1<<11);
        printf("rand_range= %d\n", rand_range);
        for (i=0; i<len; i++) {
            a[i] = rand()/(rand_range);
            b[i] = rand()/(rand_range);
            GPU_c[i] = (dtype)0;
        }
    } else {
        // here we generate random floats
        for (i=0; i<len; i++) {
          a[i] = (dtype)rand()/((dtype)RAND_MAX);
          b[i] = (dtype)rand()/((dtype)RAND_MAX);
          GPU_c[i] = (dtype)0;
        }
    }
    // ---------------- computation on CPU -------------------
    TIMER_START;
    for (i=0; i<len; i++)
      CPU_c[i] = a[i] + b[i];
    TIMER_STOP;
    //   errors[0] = 0.0;
    cputime = TIMER_ELAPSED;

    // ---------------- allocating GPU vectors -------------------
    dtype *dev_a, *dev_b, *dev_c;

    cudaMalloc(&dev_a, len*sizeof(dtype));
    cudaMalloc(&dev_b, len*sizeof(dtype));
    cudaMalloc(&dev_c, len*sizeof(dtype));


    // ===================================== EX6: Find best BLK and GRD dimensions ====================================


    int blk_size;
    float compareDimError[5][2], compareDimTime[5][2],compareDimStd[5][2], compareDimBw[5][2];  //to store the results of each block size

    for(int i_blk=5; i_blk<10; i_blk++) {   // the range of block size: 32, 64, 128, 256, 512
      
      blk_size = 1 << i_blk;

      cudaMemset(dev_a, 0, len*sizeof(dtype));
      cudaMemset(dev_b, 0, len*sizeof(dtype));
      cudaMemcpy(dev_a, a, len*sizeof(dtype), cudaMemcpyHostToDevice);
      cudaMemcpy(dev_b, b, len*sizeof(dtype), cudaMemcpyHostToDevice);

      dim3 block_size(blk_size, 1, 1);
      dim3 grid_size(GRD_SIZE, 1, 1);
      printf("block_size = %d, grid_size = %d, elements per thread = %f\n", block_size.x, grid_size.x, (float)len/(block_size.x*grid_size.x));
      
      // ------------ computation solution with Layout 1 -----------

      for (int i=-WARMUP; i<NITER; i++) {
        
        cudaMemset(dev_c, 0, len*sizeof(dtype));

        TIMER_START;
          
        Problem_solution_layout1<<<grid_size, block_size>>>(len, dev_a, dev_b, dev_c);
      
        cudaDeviceSynchronize();
        TIMER_STOP;

        gputime_iter = TIMER_ELAPSED;
        if( i >= 0) gputime_layout1[i] = gputime_iter;
      }

      
      // ----------- copy results from device to host ------------

      cudaMemcpy(GPU_c, dev_c, len*sizeof(dtype), cudaMemcpyDeviceToHost);

      // ------------- Compare GPU and CPU solution --------------

      error = 0.0f;
      for (int i = 0; i < len; i++)
        error += (float)fabs(CPU_c[i] - GPU_c[i]);

      compareDimError[i_blk-5][0] = error;
      compareDimTime[i_blk-5][0] = arithmetic_mean(gputime_layout1, NITER);
      compareDimStd[i_blk-5][0] = compute_std(gputime_layout1, NITER, compareDimTime[i_blk-5][0]);
      compareDimBw[i_blk-5][0] = (len * 3.0 * sizeof(dtype)) / (compareDimTime[i_blk-5][0] * 1e6);



      // ------------------- reset gpu buffers ---------------------

      cudaMemset(dev_a, 0, len*sizeof(dtype));
      cudaMemset(dev_b, 0, len*sizeof(dtype));
      cudaMemcpy(dev_a, a, len*sizeof(dtype), cudaMemcpyHostToDevice);
      cudaMemcpy(dev_b, b, len*sizeof(dtype), cudaMemcpyHostToDevice);


      // ------------ computation solution with Layout 3 -----------
    
      for (int i=0; i<NITER; i++) {
        cudaMemset(dev_c, 0, len*sizeof(dtype));

        TIMER_START;

        Problem_solution_layout3<<<grid_size, block_size>>>(len, dev_a, dev_b, dev_c);
        cudaDeviceSynchronize();
        TIMER_STOP;

        gputime_iter = TIMER_ELAPSED;
        if( i >= 0) gputime_layout3[i] = gputime_iter;
      }

        // ----------- copy results from device to host ------------

      cudaMemcpy(GPU_c, dev_c, len*sizeof(dtype), cudaMemcpyDeviceToHost);

      // ------------- Compare GPU and CPU solution --------------

      error = 0.0f;
      for (int i = 0; i < len; i++)
        error += (float)fabs(CPU_c[i] - GPU_c[i]);

      compareDimError[i_blk-5][1] = error;
      compareDimTime[i_blk-5][1] = arithmetic_mean(gputime_layout3, NITER);
      compareDimStd[i_blk-5][1] = compute_std(gputime_layout3, NITER, compareDimTime[i_blk-5][1]);
      compareDimBw[i_blk-5][1] = (len * 3.0 * sizeof(dtype)) / (compareDimTime[i_blk-5][1] * 1e6);

    }

    printf("\nVector len = %d, CPU time(ms) = %5.5f\n\n\n", len, cputime);
    printf("\t\tLayout1\t\t\t\t\tLayout3\n");
    // for (int j=0; j<8; j++)
    //   printf("%d\t\t", 1 << j);
    // printf("\n");
    printf("Block size\tError\tTime(Std)\tBandwidth(GB/s)\tError\tTime(Std)\tBandwidth(GB/s)\n");
    for (int i=0; i<5; i++) {
      printf("%d:\t", 1 << (i+5));

      printf("\t%4.2f\t%.3f(%.3f)\t%.3f\t", compareDimError[i][0],compareDimTime[i][0], compareDimStd[i][0],compareDimBw[i][0]);
      printf("\t%4.2f\t%.3f(%.3f)\t%.3f\t", compareDimError[i][1],compareDimTime[i][1], compareDimStd[i][1],compareDimBw[i][1]);
      printf("\n");
    }
    printf("\n");
    

    // ----------------- free GPU variable ---------------------

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // --------------------------------------------------------
  }
    
    
  return(0);

}

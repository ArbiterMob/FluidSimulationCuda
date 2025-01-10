// CUDA version - 1 cell for each thread + Shared Memory

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>

// GPU Constants
__constant__ int N;       // (WidthGrid - 2) 
__constant__ float DT;    // Instantaneous change in time (timestep)
__constant__ float VIS;   // Viscosity coefficient
__constant__ float DIFF;  // Diffusion coefficient

// CPU Global Variables
int hN = (1<<3) - 2;
float hDT = 0.016f;
float hVIS = 0.0025f;
float hDIFF = 0.1f;

// SWAP macro
#define SWAP(x0, x) {float *tmp = x0; x0 = x; x = tmp;}

// Function to check CUDA errors
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

// Function to debug a float grid
void printDebug(/*char *string,*/ float *x) {
    int i, j;

    //printf("Debugging -> %s\n", string);

    for (i = 0; i < hN + 2; i++) {
        for (j = 0; j < hN + 2; j++) {
            printf("[%f] ", x[j + i * (hN + 2)]);
        }
        printf("\n");
    }
    printf("\n\n");
}

// Function to debug and print the state of the grid
void printStateGrid(float *dens, float *u, float *v) {
    int i, j;

    printf("---------------------------------------\n");
    printf("DENSITY\n");
    for (i = 0; i < hN + 2; i++) {
        for (j = 0; j < hN + 2; j++) {
            printf("[%f] ", dens[j + i * (hN +2)]);
        }
        printf("\n");
    }
    printf("\n\n");

    printf("VELOCITY\n");
    for (i = 0; i < hN + 2; i++) {
        for (j = 0; j < hN + 2; j++) {
            printf("[%f, %f] ", u[j + i * (hN +2)], v[j + i * (hN +2)]);
        }
        printf("\n");
    }
}

// Function to measure time in seconds
double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

// CUDA device function to set the boundary conditions (borders)
__device__ void set_bndOnGPU(int b, float *d_x, int ix, int iy, int tid) {
    if (ix == 0 && iy < N + 1 && iy > 0)
        d_x[tid] = b == 1 ? -d_x[tid + 1] : d_x[tid + 1];
    else if (ix == N + 1 && iy < N + 1 && iy > 0)
        d_x[tid] = b == 1 ? -d_x[tid - 1] : d_x[tid - 1];
    else if (iy == 0 && ix > 0 && ix < N + 1)
        d_x[tid] = b == 2 ? -d_x[tid + N + 2] : d_x[tid + N + 2];
    else if (iy == N + 1 && ix > 0 && ix < N + 1)
        d_x[tid] = b == 2 ? -d_x[tid - N - 2] : d_x[tid - N - 2];
}   

// CUDA device function to set the boundary conditions (corners)
__device__ void set_crnOnGPU(int b, float *d_x, int ix, int iy, int tid) {
    if (ix == 0 && iy == 0)
        d_x[tid] = 0.5f * (d_x[tid + 1] + d_x[tid + N + 2]);
    else if (ix == 0 && iy == N + 1)
        d_x[tid] = 0.5f * (d_x[tid + 1] + d_x[tid - N - 2]);
    else if (ix == N + 1 && iy == 0)
        d_x[tid] = 0.5f * (d_x[tid - 1] + d_x[tid + N + 2]);
    else if (ix == N + 1 && iy == N + 1) 
        d_x[tid] = 0.5f * (d_x[tid - 1] + d_x[tid - N - 2]);
}

// CUDA kernel function to  ADD EXTERNAL SOURCES
__global__ void add_sourceOnGPU(float *d_x, float *d_s) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = ix + iy * (N + 2);
    int sizeShared = blockDim.x * blockDim.y;
    
    extern __shared__ float array[];
    
    float *d_x_s = (float*)array;
    float *d_s_s = (float*)&d_x_s[sizeShared];


    int sTid = threadIdx.x + threadIdx.y * blockDim.x;

    if (ix < (N + 2) && iy < (N + 2)) {
        d_x_s[sTid] = d_x[tid];
       // __syncthreads(); // to avoid bank conflicts?
        d_s_s[sTid] = d_s[tid];
        __syncthreads();

        //printf("sTid %d | tid %d | d_x_s[sTid] %f | d_s_s[sTid] %f, d_x[tid] %f | d_s[tid] %f\n",
        //    sTid, tid, d_x_s[sTid], d_s_s[sTid], d_x[tid], d_s[tid]);
        d_x_s[sTid] += DT * d_s_s[sTid];

        d_x[tid] = d_x_s[sTid];
        //__syncthreads(); // to avoid bank conflicts?
        d_s[tid] = d_s_s[sTid];

    }
}

// CUDA kernel function to perform DIFFUSION (using Jacobi iteration outside of kernel)
__global__ void diffuseOnGPU(int b, float *d_x, float *d_x0, float *d_xTemp, float alpha, float beta) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = ix + iy * (N + 2);
    int sizeShared = (blockDim.x) * (blockDim.y);
    
    extern __shared__ float array[];

    float *d_x_s = (float*)array;
    float *d_x0_s = (float*)&array[sizeShared];
    float *d_xTemp_s = (float*)&array[sizeShared * 2];

    int sTid = threadIdx.x + threadIdx.y * blockDim.x;

    if (ix < (N + 2) && iy < (N + 2)) {
        d_x_s[sTid] = d_x[tid];
        d_x0_s[sTid] = d_x0[tid];

        if (ix >= 1 && ix <= N && iy >= 1 && iy <= N) {
            d_xTemp_s[sTid] = (d_x0_s[sTid] + alpha * (d_x_s[sTid - 1] + d_x_s[sTid + 1] + 
                d_x_s[sTid - N - 2] + d_x_s[sTid + N + 2])) / beta;
            //printf("d_xTemp_s id %f in (%d, %d): d_x0_s[sTid] %f | d_x_s[sTid - 1] %f | d_x_s[sTid + 1] %f, d_x_s[sTid - N - 2] %f | d_x_s[sTid + N + 2] %f | beta %f\n",
            //    d_xTemp_s[sTid], ix, iy, d_x0_s[sTid], d_x_s[sTid - 1], d_x_s[sTid + 1], d_x_s[sTid - N - 2], d_x_s[sTid + N + 2], beta);
        }
    
        // Inefficient because multiple synchronization
        if (blockIdx.x == 0 || blockIdx.x == gridDim.x - 1 || blockIdx.y == 0 || blockIdx.y == gridDim.y - 1) {
            __syncthreads(); // synchronization intra-block to ensure that the 
                         // threads in the boundary can compute correctly
            set_bndOnGPU(b, d_xTemp, ix, iy, tid);

            __syncthreads(); // now that all the borders are complete, we can compute
                         // corners

            set_crnOnGPU(b, d_xTemp, ix, iy, tid);
        }
    }
}

// Function to simulate the evolution of density
void dens_step(dim3 grid, dim3 block, float *d_x, float *d_x0, float *d_u, float *d_v, float *d_densTemp) {
    int nBytes = sizeof(float) * block.x * block.y;
    
    add_sourceOnGPU<<<grid, block, 2 * nBytes>>>(d_x, d_x0);

    SWAP(d_x0, d_x);
}

// Function to simulate the evolution of velocity
void vel_step(dim3 grid, dim3 block, float *d_u, float *d_v, float *d_u0, float *d_v0, float *d_uTemp, float *d_vTemp) {
    int nBytes = sizeof(float) * block.x * block.y;
    
    add_sourceOnGPU<<<grid, block, 2 * nBytes>>>(d_u, d_u0);
    add_sourceOnGPU<<<grid, block>>>(d_v, d_v0);

    cudaDeviceSynchronize();

    SWAP(d_u0, d_u);
    SWAP(d_v0, d_v);

    /*
    cudaDeviceSynchronize();
    float *temp = (float*)malloc(sizeof(float) * (hN + 2) * (hN + 2));
    cudaMemcpy(temp, d_u0, sizeof(float) * (hN + 2) * (hN + 2), cudaMemcpyDeviceToHost);
    printf("u0\n");
    printDebug(temp);
    */

    float alpha = hDT * hVIS * hN * hN;
    float beta = 1 + 4 * alpha;
}

// Function to initialize the density and velocity
void initializeParameters(float *dens, float *dens_prev, float *u, float *u_prev, float *v, float *v_prev) {
    int i, j;
    int center_x = (hN + 2) / 2, center_y = (hN + 2) / 2;
    int radius = (hN + 2) / 8;

    // density source
    for (i = 0; i < hN + 2; i++) {
        for (j = 0; j < hN + 2; j++) {
            if ((j < center_x + radius) && (j >= center_x - radius) &&
                (i < center_y + radius) && (i >= center_y - radius))
                dens_prev[j + i * (hN + 2)] = (rand() % 100) / 1000.0f;
            else 
                dens_prev[j + i * (hN + 2)] = 0.0f;
            dens[j + i * (hN + 2)] = 0.0f;
        }
    } 

    // velocity
    for (i = 0; i < hN + 2; i++) {
        for (j = 0; j < hN + 2; j++) {
            u_prev[j + i * (hN +2)] = (rand() % 100) / 100.0f;
            v_prev[j + i * (hN +2)] = (rand() % 100) / 100.0f;

            u[j + i * (hN +2)] = 0.0f;
            v[j + i * (hN +2)] = 0.0f;
        }
    }
}

int main(int argc, char **argv) {
     // Check command line arguments
    if (argc != 3) {
        printf("Usage: %s <block_dim_x> <block_dim_y>\n", argv[0]);
        return 1;
    }
    
    // Set CUDA device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Parse command line arguments
    int block_dim_x = atoi(argv[1]);
    int block_dim_y = atoi(argv[2]);

    int size = (hN + 2) * (hN + 2);
    int nBytes = size * sizeof(float);
    double iStart, iElaps;

    // Allocate host memory
    float *u, *u_prev, *v, *v_prev, *dens, *dens_prev;
    u = (float*)malloc(nBytes);
    u_prev = (float*)malloc(nBytes);
    v = (float*)malloc(nBytes);
    v_prev = (float*)malloc(nBytes);
    dens = (float*)malloc(nBytes);
    dens_prev = (float*)malloc(nBytes);
    
    // Allocate device memory
    float *d_u, *d_u_prev, *d_v, *d_v_prev, *d_dens, *d_dens_prev, *d_uTemp, *d_vTemp, *d_densTemp;
    CHECK(cudaMalloc((void **)&d_u, nBytes));
    CHECK(cudaMalloc((void **)&d_u_prev, nBytes));
    CHECK(cudaMalloc((void **)&d_v, nBytes));
    CHECK(cudaMalloc((void **)&d_v_prev, nBytes));
    CHECK(cudaMalloc((void **)&d_dens, nBytes));
    CHECK(cudaMalloc((void **)&d_dens_prev, nBytes));
    CHECK(cudaMalloc((void **)&d_uTemp, nBytes));
    CHECK(cudaMalloc((void **)&d_vTemp, nBytes));
    CHECK(cudaMalloc((void **)&d_densTemp, nBytes));

    CHECK(cudaMemcpyToSymbol(N, &hN, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(DT, &hDT, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(VIS, &hVIS, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(DIFF, &hDIFF, sizeof(float)));

    // Define grid and block dimensions
    dim3 block, grid;
    block = dim3(block_dim_x, block_dim_y);
    grid = dim3(((hN + 2) + block_dim_x - 1) / block_dim_x, ((hN + 2) + block_dim_y - 1) / block_dim_y);

    // Simulation
    int z = 0;
    int first = 1;
    iStart = cpuSecond();
    while (z++ < 1) {
        if (first) {
            initializeParameters(dens, dens_prev, u, u_prev, v, v_prev);
            first = 0;
            //printf("HELLO init\n");
            //printStateGrid(dens_prev, u_prev, v_prev);
            //printf("Hello end\n");

            CHECK(cudaMemcpy(d_u, u, nBytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_u_prev, u_prev, nBytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_v, v, nBytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_v_prev, v_prev, nBytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_dens, dens, nBytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_dens_prev, dens_prev, nBytes, cudaMemcpyHostToDevice));
        } else {
            for (int i = 0; i < size; i++) {
                u_prev[i] = 0.0f;
                v_prev[i] = 0.0f;
                dens_prev[i] = 0.0f;
            }

            CHECK(cudaMemcpy(d_u_prev, u_prev, nBytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_v_prev, v_prev, nBytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_dens_prev, dens_prev, nBytes, cudaMemcpyHostToDevice));
        }

        vel_step(grid, block, d_u, d_v, d_u_prev, d_v_prev, d_uTemp, d_vTemp);
        dens_step(grid, block, d_dens, d_dens_prev, d_u, d_v, d_densTemp);

        // DA METTERE NEL CICLO SOLO PER DEBUG 
        /*
        CHECK(cudaMemcpy(u, d_u, nBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(u_prev, d_u_prev, nBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(v, d_v, nBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(v_prev, d_v_prev, nBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(dens, d_dens, nBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(dens_prev, d_dens_prev, nBytes, cudaMemcpyDeviceToHost));
        printStateGrid(dens, u, v);
        */     
    }
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("grid: %d, <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", size, grid.x, grid.y, block.x, block.y, iElaps);

    CHECK(cudaMemcpy(u, d_u, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(v, d_v, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(dens, d_dens, nBytes, cudaMemcpyDeviceToHost));
    printStateGrid(dens, u, v);        

    // Cleaning
    free(u); free(u_prev);
    free(v); free(v_prev);
    free(dens); free(dens_prev);

    CHECK(cudaFree(d_u)); CHECK(cudaFree(d_u_prev));
    CHECK(cudaFree(d_v)); CHECK(cudaFree(d_v_prev));
    CHECK(cudaFree(d_dens)); CHECK(cudaFree(d_dens_prev));
    CHECK(cudaFree(d_uTemp)); CHECK(cudaFree(d_vTemp)); CHECK(cudaFree(d_densTemp));

    return 0;
}
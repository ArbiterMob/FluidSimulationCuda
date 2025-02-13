// CUDA first version naive (multiple cells for each thread)
// Using Shared Memory

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>

// GPU Constants
__constant__ int N;         // (WidthGrid - 2) 
__constant__ float DT;      // Instantaneous change in time (timestep)
__constant__ float VIS;     // Viscosity coefficient
__constant__ float DIFF;    // Diffusion coefficient
__constant__ int RADIUS;    // Radius for halo region 

// CPU Global Variables
int hN = (1<<13) - 2;
float hDT = 0.016f;
float hVIS = 0.0025f;
float hDIFF = 0.1f;
int hRADIUS = 1;
constexpr int GRID_DIVISION_FACTOR = 2; // every thread computes 1 << GRID_DIVISION_FACTOR cells

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

// CUDA device function to compute borders and corner
__device__ void setBordersOnGPU(int b, float *x, int ix, int iy, int start_x, int start_y, int section_size_x, int section_size_y) {
    int i, j, nIX, nIY, nTid;

    //printf("start_x %d | start_y %d\nDEBUG - %d\n\n", start_x, start_y, start_y == 0 && start_x == ((blockDim.x - 1) + blockIdx.x * blockDim.x) * section_size_x);

    if (ix == 0 && iy == 0) {
        for (i = 0; i < section_size_y; i++) {
            nIY = i;
            nTid = i * (N + 2);
            set_bndOnGPU(b, x, 0, nIY, nTid);
        }
        for (j = 0; j < section_size_x; j++) {
            nIX = j;
            nTid = nIX;
            set_bndOnGPU(b, x, nIX, 0, nTid);
        }
        set_crnOnGPU(b, x, 0, 0, 0); 
    } else if (ix == blockDim.x * gridDim.x - 1 && iy == blockDim.y * gridDim.y - 1) {
        for (i = 0; i < section_size_y; i++) {
            nIY = start_y + i;
            nTid = (N + 1) + nIY * (N + 2);
            set_bndOnGPU(b, x, N + 1, nIY, nTid);
        }
        for (j = 0; j < section_size_x; j++) {
            nIX = start_x + j;
            nTid = nIX + (N + 1) * (N + 2);
            set_bndOnGPU(b, x, nIX, N + 1, nTid);
        }
        set_crnOnGPU(b, x, N + 1, N + 1, (N + 1) + (N + 1) * (N + 2));
    } else if (iy == 0 && ix == blockDim.x * gridDim.x - 1) {
        for (j = 0; j < section_size_x; j++) {
            nIX = start_x + j;
            nTid = nIX;
            set_bndOnGPU(b, x, nIX, 0, nTid);
        }
        for (i = 0; i < section_size_y; i++) {
            nIY = start_y + i;
            nTid = (N + 1) + nIY * (N + 2);
            set_bndOnGPU(b, x, N + 1, nIY, nTid);
        }
        set_crnOnGPU(b, x, N + 1, 0, N + 1);
    } else if (ix == 0 && iy == blockDim.y * gridDim.y - 1) {
        for (i = 0; i < section_size_y; i++) {
            nIY = start_y + i;
            nTid = nIY * (N + 2);
            set_bndOnGPU(b, x, 0, nIY, nTid);
        }
        for (j = 0; j < section_size_x; j++) {
            nIX = start_x + j;
            nTid = nIX + (N + 1) * (N + 2);
            set_bndOnGPU(b, x, nIX, N + 1, nTid);
        }
        set_crnOnGPU(b, x, 0, N + 1, (N + 1) * (N + 2));
    } else if (ix == 0) {
        for (i = 0; i < section_size_y; i++) {
            nIY = start_y + i;
            nTid = nIY * (N + 2);
            set_bndOnGPU(b, x, 0, nIY, nTid);
        }
    } else if (iy == 0) {
        for (j = 0; j < section_size_x; j++) {
            nIX = start_x + j;
            nTid = nIX;
            set_bndOnGPU(b, x, nIX, 0, nTid);
        }
    } else if (ix == blockDim.x * gridDim.x - 1) {
        for (i = 0; i < section_size_y; i++) {
            nIY = start_y + i;
            nTid = (N + 1) + nIY * (N + 2);
            set_bndOnGPU(b, x, N + 1, nIY, nTid);
        }
    } else if (iy == blockDim.y * gridDim.y - 1) {
        for (j = 0; j < section_size_x; j++) {
            nIX = start_x + j;
            nTid = nIX + (N + 1) * (N + 2);
            set_bndOnGPU(b, x, nIX, N + 1, nTid);
        }
    }
}

template<int UNROLL_FACTOR>
// CUDA kernel function to  ADD EXTERNAL SOURCES
__global__ void add_sourceOnGPU(float *d_x, float *d_s) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int lTid = threadIdx.x + threadIdx.y * blockDim.x;
    int i, j, nIX, nIY, nTid, sTid;

    int size = (N + 2);
    int section_size_x = (size - 1) / (blockDim.x * gridDim.x) + 1;
    int section_size_y = (size - 1) / (blockDim.y * gridDim.y) + 1;
    int sizeShared = section_size_x * section_size_y * blockDim.x * blockDim.y;

    int start_x = ix * section_size_x;
    int start_y = iy * section_size_y;
    //printf("ix %d | iy %d | section_size_x %d | section_size_y %d | start_x %d | start_y %d\n",
    //    ix, iy, section_size_x, section_size_y, start_x, start_y);

    extern __shared__ float sharedMem[];
    float *d_x_s = (float*)sharedMem;
    float *d_s_s = (float*)&sharedMem[sizeShared];

    // Unrolled section processing
    if (start_y + (UNROLL_FACTOR - 1) < size) {
        #pragma unroll
        for (i = 0; i < UNROLL_FACTOR; i++) {
            if (start_x + (UNROLL_FACTOR - 1) < size) {
                #pragma unroll
                for (j = 0; j < UNROLL_FACTOR; j++) {
                    nIX = start_x + j;
                    nIY = start_y + i;
                    nTid = nIX + nIY * (N + 2);
                    sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

                    if (nIX < (N + 2) && nIY < (N + 2)) {
                        d_x_s[sTid] = d_x[nTid];
                        d_s_s[sTid] = d_s[nTid];
                    }
                }
            }
        }
    }

    // Handle remaining elements
    for (i = UNROLL_FACTOR; i < section_size_y && start_y + i < size; i++) {
        for (j = UNROLL_FACTOR; j < section_size_x && start_x + j < size; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);
            sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

            d_x_s[sTid] = d_x[nTid];
            d_s_s[sTid] = d_s[nTid];
        }
    }

    __syncthreads();

    // Unrolled section processing
    if (start_y + (UNROLL_FACTOR - 1) < size) {
        #pragma unroll
        for (i = 0; i < UNROLL_FACTOR; i++) {
            if (start_x + (UNROLL_FACTOR - 1) < size) {
                #pragma unroll
                for (j = 0; j < UNROLL_FACTOR; j++) {
                    nIX = start_x + j;
                    nIY = start_y + i;
                    nTid = nIX + nIY * (N + 2);
                    sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

                    if (nIX < (N + 2) && nIY < (N + 2)) {
                        d_x[nTid] = d_x_s[sTid] + DT * d_s_s[sTid];
                    }
                }
            }
        }
    }

    // Handle remaining elements
    for (i = UNROLL_FACTOR; i < section_size_y && start_y + i < size; i++) {
        for (j = UNROLL_FACTOR; j < section_size_x && start_x + j < size; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);
            sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

            d_x[nTid] = d_x_s[sTid] + DT * d_s_s[sTid];
        }
    }
}

template <int UNROLL_FACTOR>
// CUDA kernel function to perform DIFFUSION (using Jacobi iteration outside of kernel)
__global__ void diffuseOnGPU(int b, float *d_x, float *d_x0, float *d_xTemp, float alpha, float beta) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int lTid = threadIdx.x + threadIdx.y * blockDim.x;
    int i, j, nIX, nIY, nTid, e_lx, e_ly, sTid;

    int size = (N + 2);
    int section_size_x = (size - 1) / (blockDim.x * gridDim.x) + 1;
    int section_size_y = (size - 1) / (blockDim.y * gridDim.y) + 1;
    int sizeShared = section_size_x * section_size_y * blockDim.x * blockDim.y;

    int start_x = ix * section_size_x;
    int start_y = iy * section_size_y;

    extern __shared__ float sharedMem[];
    float *d_x0_s = (float*)sharedMem;
    float *d_x_s = (float*)&sharedMem[sizeShared];

    // Unrolled section processing
    if (start_y + (UNROLL_FACTOR - 1) < size) {
        #pragma unroll
        for (i = 0; i < UNROLL_FACTOR; i++) {
            if (start_x + (UNROLL_FACTOR - 1) < size) {
                #pragma unroll
                for (j = 0; j < UNROLL_FACTOR; j++) {
                    nIX = start_x + j;
                    nIY = start_y + i;
                    nTid = nIX + nIY * (N + 2);
                    sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

                    e_lx = j + RADIUS;
                    e_ly = i + RADIUS;

                    if (nIX < (N + 2) && nIY < (N + 2)) {
                        if (i < RADIUS) { // copy top and bottom halo
                            //Copy Top Halo Element
		                    if (iy > 0) // Boundary Check 
                                d_x_s[e_lx + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = d_x[nTid - RADIUS * (N + 2)];
		        
                            //Copy Bottom Halo Element
		                    if (iy < blockDim.y * gridDim.y - 1) // Boundary Check
                                d_x_s[e_lx + (e_ly + section_size_y) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = d_x[nTid + section_size_y * (N + 2)];
                        }  

                        if (j < RADIUS) { // copy left and right halo
                            // Copy Left Halo Element
                            if (ix > 0) // Boundary Check
                                d_x_s[j + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = d_x[nTid - RADIUS];
        
                            // Copy Right Halo Element
                            if (ix < blockDim.x * gridDim.x - 1) // Boundary Check
                                d_x_s[(e_lx + section_size_x) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = d_x[nTid + section_size_x];
                        }

                        d_x_s[e_lx + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = d_x[nTid];
                        d_x0_s[sTid] = d_x0[nTid];
                    }   
                }
            }
        }
    }

    // Handle remaining elements
    for (i = UNROLL_FACTOR; i < section_size_y && start_y + i < size; i++) {
        for (j = UNROLL_FACTOR; j < section_size_x && start_x + j < size; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);
            sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);
        
            e_lx = j + RADIUS;
            e_ly = i + RADIUS;

            if (i < RADIUS) { // copy top and bottom halo
                //Copy Top Halo Element
		        if (iy > 0) // Boundary Check 
                    d_x_s[e_lx + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = d_x[nTid - RADIUS * (N + 2)];
		        
                //Copy Bottom Halo Element
		        if (iy < blockDim.y * gridDim.y - 1) // Boundary Check
                    d_x_s[e_lx + (e_ly + section_size_y) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = d_x[nTid + section_size_y * (N + 2)];
            }  

            if (j < RADIUS) { // copy left and right halo
                // Copy Left Halo Element
                if (ix > 0) // Boundary Check
                    d_x_s[j + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = d_x[nTid - RADIUS];
        
                // Copy Right Halo Element
                if (ix < blockDim.x * gridDim.x - 1) // Boundary Check
                    d_x_s[(e_lx + section_size_x) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = d_x[nTid + section_size_x];
            }

            d_x_s[e_lx + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = d_x[nTid];
            d_x0_s[sTid] = d_x0[nTid];  
        }
    }

    __syncthreads();
    /*
    for (i = 0; i < section_size_y; i++) {
        for (j = 0; j < section_size_x; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);
            sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

            e_lx = j + RADIUS;
            e_ly = i + RADIUS;
            printf("OLD nIndex(%d %d) | sTid %d | d_x_s %f | d_x %f | d_x0_s %f | d_x0 %f\n",
                nIX, nIY, sTid, d_x_s[e_lx + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], d_x[nTid], d_x0_s[sTid], d_x0[nTid]);
        }
    }
    */

    // Unrolled section processing
    if (start_y + (UNROLL_FACTOR - 1) < size) {
        #pragma unroll
        for (i = 0; i < UNROLL_FACTOR; i++) {
            if (start_x + (UNROLL_FACTOR - 1) < size) {
                #pragma unroll
                for (j = 0; j < UNROLL_FACTOR; j++) {
                    nIX = start_x + j;
                    nIY = start_y + i;
                    nTid = nIX + nIY * (N + 2);
                    sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

                    e_lx = j + RADIUS;
                    e_ly = i + RADIUS;

                    if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) {
                        d_xTemp[nTid] = (d_x0_s[sTid] + alpha * (d_x_s[e_lx + (e_ly + 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] + d_x_s[e_lx + (e_ly - 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] +
                            d_x_s[(e_lx + 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] + d_x_s[(e_lx - 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))])) / beta;
                        //printf("nIndex(%d %d) start(%d %d) | sTid %d | d_xTemp %f | d_x0[tid] %f | d_x[tid - 1] %f - %f| d_x[tid + 1] %f - %f| d_x[tid - N - 2] %f - %f | d_x[tid + N + 2] %f - %f| beta %f\n",
                        //    nIX, nIY, start_x, start_y, sTid, d_xTemp[nTid], d_x0_s[sTid], d_x_s[(e_lx - 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], d_x[nTid - 1], d_x_s[(e_lx + 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], d_x[nTid + 1],
                        //    d_x_s[e_lx + (e_ly - 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], d_x[nTid - 1 * (N + 2)], d_x_s[e_lx + (e_ly + 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], d_x[nTid + 1 * (N + 2)], beta);
                    }
                }
            }
        }
    }

    // Handle remaining elements
    for (i = UNROLL_FACTOR; i < section_size_y && start_y + i < size; i++) {
        for (j = UNROLL_FACTOR; j < section_size_x && start_x + j < size; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);
            sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

            e_lx = j + RADIUS;
            e_ly = i + RADIUS;
            
            if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) {
                d_xTemp[nTid] = (d_x0_s[sTid] + alpha * (d_x_s[e_lx + (e_ly + 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] + d_x_s[e_lx + (e_ly - 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] +
                    d_x_s[(e_lx + 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] + d_x_s[(e_lx - 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))])) / beta;
                //printf("nIndex(%d %d) start(%d %d) | sTid %d | d_xTemp %f | d_x0[tid] %f | d_x[tid - 1] %f - %f| d_x[tid + 1] %f - %f| d_x[tid - N - 2] %f - %f | d_x[tid + N + 2] %f - %f| beta %f\n",
                //    nIX, nIY, start_x, start_y, sTid, d_xTemp[nTid], d_x0_s[sTid], d_x_s[(e_lx - 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], d_x[nTid - 1], d_x_s[(e_lx + 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], d_x[nTid + 1],
                //    d_x_s[e_lx + (e_ly - 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], d_x[nTid - 1 * (N + 2)], d_x_s[e_lx + (e_ly + 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], d_x[nTid + 1 * (N + 2)], beta);
            }
        }
    }

    // maybe this is less performant ...
    setBordersOnGPU(b, d_xTemp, ix, iy, start_x, start_y, section_size_x, section_size_y);
}

template <int UNROLL_FACTOR>
// CUDA kernel function to perform ADVECTION (using bilinear interpolation)
__global__ void advectOnGPU(int b, float *d_d, float *d_d0, float *d_u, float *d_v) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int lTid = threadIdx.x + threadIdx.y * blockDim.x;
    int i, j, nIX, nIY, nTid, sTid;

    int size = (N + 2);
    int section_size_x = (size - 1) / (blockDim.x * gridDim.x) + 1;
    int section_size_y = (size - 1) / (blockDim.y * gridDim.y) + 1;
    int sizeShared = section_size_x * section_size_y * blockDim.x * blockDim.y;

    int start_x = ix * section_size_x;
    int start_y = iy * section_size_y;;

    extern __shared__ float sharedMem[];
    float *d_d0_s = (float*)sharedMem;

    int i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    dt0 = DT * N;
    // Unrolled section processing
    if (start_y + (UNROLL_FACTOR - 1) < size) {
        #pragma unroll
        for (i = 0; i < UNROLL_FACTOR; i++) {
            if (start_x + (UNROLL_FACTOR - 1) < size) {
                #pragma unroll
                for (j = 0; j < UNROLL_FACTOR; j++) {
                    nIX = start_x + j;
                    nIY = start_y + i;
                    nTid = nIX + nIY * (N + 2);
                    sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

                    if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) {
                        x = nIX - dt0 * d_u[nTid];
                        y = nIY - dt0 * d_v[nTid];

                        if (x < 0.5) 
                            x = 0.5; 
                        if (x > N + 0.5) 
                            x = N + 0.5;
                        j0 = (int)x;
                        j1 = j0 + 1;

                        if (y < 0.5) 
                            y = 0.5; 
                        if (y > N + 0.5) 
                            y = N + 0.5; 
                        i0 = (int)y;
                        i1 = i0 + 1;

                        s1 = x - j0;
                        s0 = 1 - s1;
                        t1 = y - i0;
                        t0 = 1 - t1;

                        d_d0_s[sTid] = s0 * t0 * d_d0[j0 + i0 * (N + 2)];
                        d_d0_s[sTid + sizeShared] = s0 * t1 * d_d0[j0 + i1 * (N + 2)];
                        d_d0_s[sTid + 2 * sizeShared] = s1 * t0 * d_d0[j1 + i0 * (N + 2)];
                        d_d0_s[sTid + 3 * sizeShared] = s1 * t1 * d_d0[j1 + i1 * (N + 2)];

                        //__syncthreads();
                        //printf("sTid %d %d %d %d| lTid %d (%d %d) | d_d0 1 %f %f| d_d0 2 %f %f| d_d0 3 %f %f | d_d0 4 %f %f\n",
                        //    sTid, sTid + sizeShared, sTid + 2 * sizeShared, sTid + 3 * sizeShared, lTid, nIX, nIY, d_d0_s[sTid], d_d0[j0 + i0 * (N + 2)], d_d0_s[sTid + sizeShared], d_d0[j0 + i1 * (N + 2)],
                        //    d_d0_s[sTid + 2 * sizeShared], d_d0[j1 + i0 * (N + 2)], d_d0_s[sTid + 3 * sizeShared], d_d0[j1 + i1 * (N + 2)]);
                    }
                }
            }
        }
    }

    // Handle remaining elements
    for (i = UNROLL_FACTOR; i < section_size_y && start_y + i < size; i++) {
        for (j = UNROLL_FACTOR; j < section_size_x && start_x + j < size; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);
            sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

            if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) {
                x = nIX - dt0 * d_u[nTid];
                y = nIY - dt0 * d_v[nTid];

                if (x < 0.5) 
                    x = 0.5; 
                if (x > N + 0.5) 
                    x = N + 0.5;
                j0 = (int)x;
                j1 = j0 + 1;

                if (y < 0.5) 
                    y = 0.5; 
                if (y > N + 0.5) 
                    y = N + 0.5; 
                i0 = (int)y;
                i1 = i0 + 1;

                s1 = x - j0;
                s0 = 1 - s1;
                t1 = y - i0;
                t0 = 1 - t1;

                d_d0_s[sTid] = s0 * t0 * d_d0[j0 + i0 * (N + 2)];
                d_d0_s[sTid + sizeShared] = s0 * t1 * d_d0[j0 + i1 * (N + 2)];
                d_d0_s[sTid + 2 * sizeShared] = s1 * t0 * d_d0[j1 + i0 * (N + 2)];
                d_d0_s[sTid + 3 * sizeShared] = s1 * t1 * d_d0[j1 + i1 * (N + 2)];
            }
        }
    }

    __syncthreads();
    /*
    for (i = 0; i < section_size_y; i++) {
        for (j = 0; j < section_size_x; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);
            sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

            if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) 
                printf("sTid %d %d %d %d| lTid %d (%d %d) | d_d0 1 %f | d_d0 2 %f | d_d0 3 %f | d_d0 4 %f\n",
                    sTid, sTid + sizeShared, sTid + 2 * sizeShared, sTid + 3 * sizeShared, lTid, nIX, nIY, d_d0_s[sTid], d_d0_s[sTid + sizeShared],
                    d_d0_s[sTid + 2 * sizeShared], d_d0_s[sTid + 3 * sizeShared]);

        }
    }
    __syncthreads();
    */    

    // Unrolled section processing
    if (start_y + (UNROLL_FACTOR - 1) < size) {
        #pragma unroll
        for (i = 0; i < UNROLL_FACTOR; i++) {
            if (start_x + (UNROLL_FACTOR - 1) < size) {
                #pragma unroll
                for (j = 0; j < UNROLL_FACTOR; j++) { 
                    nIX = start_x + j;
                    nIY = start_y + i;
                    nTid = nIX + nIY * (N + 2);
                    sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

                    if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) {
                        d_d[nTid] = d_d0_s[sTid] + d_d0_s[sTid + sizeShared] + d_d0_s[sTid + 2 * sizeShared] + d_d0_s[sTid + 3 * sizeShared];
                        //printf("NEW sTid %d %d %d %d| lTid %d (%d %d) | d_d %f | d_d0 1 %f | d_d0 2 %f | d_d0 3 %f | d_d0 4 %f\n",
                        //    sTid, sTid + sizeShared, sTid + 2 * sizeShared, sTid + 3 * sizeShared, lTid, nIX, nIY, d_d[nTid], d_d0_s[sTid], d_d0_s[sTid + sizeShared],
                        //    d_d0_s[sTid + 2 * sizeShared], d_d0_s[sTid + 3 * sizeShared]);
                    } 
                }
            }
        }
    }
    
    // Handle remaining elements
    for (i = UNROLL_FACTOR; i < section_size_y && start_y + i < size; i++) {
        for (j = UNROLL_FACTOR; j < section_size_x && start_x + j < size; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);
            sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

            if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) {
                d_d[nTid] = d_d0_s[sTid] + d_d0_s[sTid + sizeShared] + d_d0_s[sTid + 2 * sizeShared] + d_d0_s[sTid + 3 * sizeShared];
                //printf("NEW sTid %d %d %d %d| lTid %d (%d %d) | d_d %f | d_d0 1 %f | d_d0 2 %f | d_d0 3 %f | d_d0 4 %f\n",
                //    sTid, sTid + sizeShared, sTid + 2 * sizeShared, sTid + 3 * sizeShared, lTid, nIX, nIY, d_d[nTid], d_d0_s[sTid], d_d0_s[sTid + sizeShared],
                //    d_d0_s[sTid + 2 * sizeShared], d_d0_s[sTid + 3 * sizeShared]);
            }
        }
    }

    // maybe this is less performant ...
    setBordersOnGPU(b, d_d, ix, iy, start_x, start_y, section_size_x, section_size_y);
    
    /*
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int i, j, nIX, nIY, nTid;

    int size = (N + 2);
    int section_size_x = (size - 1) / (blockDim.x * gridDim.x) + 1;
    int section_size_y = (size - 1) / (blockDim.y * gridDim.y) + 1;

    int start_x = ix * section_size_x;
    int start_y = iy * section_size_y;

    int i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    dt0 = DT * N;
    // Unrolled section processing
    if (start_y + (UNROLL_FACTOR - 1) < size) {
        #pragma unroll
        for (i = 0; i < UNROLL_FACTOR; i++) {
            if (start_x + (UNROLL_FACTOR - 1) < size) {
                #pragma unroll
                for (j = 0; j < UNROLL_FACTOR; j++) {
                    nIX = start_x + j;
                    nIY = start_y + i;
                    nTid = nIX + nIY * (N + 2);
                    if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) {
                        x = nIX - dt0 * d_u[nTid];
                        y = nIY - dt0 * d_v[nTid];

                        if (x < 0.5) 
                            x = 0.5; 
                        if (x > N + 0.5) 
                            x = N + 0.5;
                        j0 = (int)x;
                        j1 = j0 + 1;

                        if (y < 0.5) 
                            y = 0.5; 
                        if (y > N + 0.5) 
                            y = N + 0.5; 
                        i0 = (int)y;
                        i1 = i0 + 1;

                        s1 = x - j0;
                        s0 = 1 - s1;
                        t1 = y - i0;
                        t0 = 1 - t1;

                        d_d[nTid] = s0 * (t0 * d_d0[j0 + i0 * (N + 2)] + t1 * d_d0[j0 + i1 * (N + 2)]) +
                            s1 * (t0 * d_d0[j1 + i0 * (N + 2)] + t1 * d_d0[j1 + i1 * (N + 2)]);
                        //printf("(%f, %f) | d_d[nTid] %f (%d, %d)| d_d0[j0 + i0 * (N + 2)] %f | d_d0[j0 + i1 * (N + 2)] %f | d_d0[j1 + i0 * (N + 2)] %f | d_d0[j1 + i1 * (N + 2)] %f\n",
                        //    x, y, d_d[nTid], nIX, nIY, d_d0[j0 + i0 * (N + 2)], d_d0[j0 + i1 * (N + 2)], d_d0[j1 + i0 * (N + 2)], d_d0[j1 + i1 * (N + 2)]);
                    }
                }
            }
        }
    }

    // Handle remaining elements
    for (i = UNROLL_FACTOR; i < section_size_y && start_y + i < size; i++) {
        for (j = UNROLL_FACTOR; j < section_size_x && start_x + j < size; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);
            if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) {
                x = nIX - dt0 * d_u[nTid];
                y = nIY - dt0 * d_v[nTid];

                if (x < 0.5) 
                    x = 0.5; 
                if (x > N + 0.5) 
                    x = N + 0.5;
                j0 = (int)x;
                j1 = j0 + 1;

                if (y < 0.5) 
                    y = 0.5; 
                if (y > N + 0.5) 
                    y = N + 0.5; 
                i0 = (int)y;
                i1 = i0 + 1;

                s1 = x - j0;
                s0 = 1 - s1;
                t1 = y - i0;
                t0 = 1 - t1;

                d_d[nTid] = s0 * (t0 * d_d0[j0 + i0 * (N + 2)] + t1 * d_d0[j0 + i1 * (N + 2)]) +
                    s1 * (t0 * d_d0[j1 + i0 * (N + 2)] + t1 * d_d0[j1 + i1 * (N + 2)]);
                //printf("(%f, %f) | d_d[nTid] %f (%d, %d)| d_d0[j0 + i0 * (N + 2)] %f | d_d0[j0 + i1 * (N + 2)] %f | d_d0[j1 + i0 * (N + 2)] %f | d_d0[j1 + i1 * (N + 2)] %f\n",
                //    x, y, d_d[nTid], nIX, nIY, d_d0[j0 + i0 * (N + 2)], d_d0[j0 + i1 * (N + 2)], d_d0[j1 + i0 * (N + 2)], d_d0[j1 + i1 * (N + 2)]);
            }
        }
    }

    // maybe this is less performant ...
    setBordersOnGPU(b, d_d, ix, iy, start_x, start_y, section_size_x, section_size_y);
    */
}

template <int UNROLL_FACTOR>
// CUDA kernel function to COMPUTE DIVERGENCE AND PRESSURE
__global__ void computeDivergenceAndPressureOnGPU(float *d_u, float *d_v, float *p, float *div) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int lTid = threadIdx.x + threadIdx.y * blockDim.x;
    int i, j, nIX, nIY, nTid, e_lx, e_ly;

    int size = (N + 2);
    int section_size_x = (size - 1) / (blockDim.x * gridDim.x) + 1;
    int section_size_y = (size - 1) / (blockDim.y * gridDim.y) + 1;
    //int sizeShared = section_size_x * section_size_y * blockDim.x * blockDim.y;
    int sizeSharedU = (section_size_x + 2) * section_size_y * blockDim.x * blockDim.y;

    int start_x = ix * section_size_x;
    int start_y = iy * section_size_y;

    extern __shared__ float sharedMem[];
    float *d_u_s = (float*)sharedMem;
    float *d_v_s = (float*)&sharedMem[sizeSharedU];

    float h = 1.0f / N;
    // Unrolled section processing
    if (start_y + (UNROLL_FACTOR - 1) < size) {
        #pragma unroll
        for (i = 0; i < UNROLL_FACTOR; i++) {
            if (start_x + (UNROLL_FACTOR - 1) < size) {
                #pragma unroll
                for (j = 0; j < UNROLL_FACTOR; j++) {
                    nIX = start_x + j;
                    nIY = start_y + i;
                    nTid = nIX + nIY * (N + 2);
                    //sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

                    e_lx = j + RADIUS;
                    e_ly = i + RADIUS;

                    if (ix < (N + 2) && iy < (N + 2)) {
                        if (i < RADIUS) { // copy top and bottom halo
                            //Copy Top Halo Element
		                    if (iy > 0) // Boundary Check
                                d_v_s[j + i * (section_size_x* RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))] = d_v[nTid - RADIUS * (N + 2)];

		                    //Copy Bottom Halo Element
		                    if (iy < blockDim.y * gridDim.y - 1) // Boundary Check
                                d_v_s[j + (e_ly + section_size_y) * (section_size_x * RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))] = d_v[nTid + section_size_y * (N + 2)];
                        }  

                        if (j < RADIUS) { // copy left and right halo
                            // Copy Left Halo Element
                            if (ix > 0) // Boundary Check
                                d_u_s[j + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))] = d_u[nTid - RADIUS];
        
                            // Copy Right Halo Element
                            if (ix < blockDim.x * gridDim.x - 1) // Boundary Check
                                d_u_s[(e_lx + section_size_x) + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))] = d_u[nTid + section_size_x];
                        }

                        // Copy Current Location
                        d_v_s[j + e_ly * (section_size_x * RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))] = d_v[nTid];
                        d_u_s[e_lx + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))] = d_u[nTid];
                    }
                }
            }
        }
    }

    // Handle remaining elements
    for (i = UNROLL_FACTOR; i < section_size_y && start_y + i < size; i++) {
        for (j = UNROLL_FACTOR; j < section_size_x && start_x + j < size; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);
            //sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

            e_lx = j + RADIUS;
            e_ly = i + RADIUS;

            if (ix < (N + 2) && iy < (N + 2)) {
                if (i < RADIUS) { // copy top and bottom halo
                    //Copy Top Halo Element
		            if (iy > 0) // Boundary Check
                        d_v_s[j + i * (section_size_x* RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))] = d_v[nTid - RADIUS * (N + 2)];

		            //Copy Bottom Halo Element
		            if (iy < blockDim.y * gridDim.y - 1) // Boundary Check
                        d_v_s[j + (e_ly + section_size_y) * (section_size_x * RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))] = d_v[nTid + section_size_y * (N + 2)];
                }  

                if (j < RADIUS) { // copy left and right halo
                    // Copy Left Halo Element
                    if (ix > 0) // Boundary Check
                        d_u_s[j + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))] = d_u[nTid - RADIUS];
        
                    // Copy Right Halo Element
                    if (ix < blockDim.x * gridDim.x - 1) // Boundary Check
                        d_u_s[(e_lx + section_size_x) + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))] = d_u[nTid + section_size_x];
                }

                // Copy Current Location
                d_v_s[j + e_ly * (section_size_x * RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))] = d_v[nTid];
                d_u_s[e_lx + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))] = d_u[nTid];
            }
        }
    }

    __syncthreads();
    /*
    for (i = 0; i < section_size_y; i++) {
        for (j = 0; j < section_size_x; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);
            //sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

            e_lx = j + RADIUS;
            e_ly = i + RADIUS;
            //printf("sTid %d lTid %d nIndex(%d %d) | d_v_s %f | d_v %f | d_u_s %f | d_u %f\n",
            //    sTid, lTid, nIX, nIY, d_v_s[j + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))], d_v[nTid], d_u_s[e_lx + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))], d_u[nTid]);
        }
    }
    */

    // Unrolled section processing
    if (start_y + (UNROLL_FACTOR - 1) < size) {
        #pragma unroll
        for (i = 0; i < UNROLL_FACTOR; i++) {
            if (start_x + (UNROLL_FACTOR - 1) < size) {
                #pragma unroll
                for (j = 0; j < UNROLL_FACTOR; j++) {
                    nIX = start_x + j;
                    nIY = start_y + i;
                    nTid = nIX + nIY * (N + 2);

                    e_lx = j + RADIUS;
                    e_ly = i + RADIUS;

                    if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) {
                        div[nTid] = -0.5f * h * (d_u_s[(e_lx + 1) + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))] - d_u_s[(e_lx - 1) + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))]
                            + d_v_s[j + (e_ly + 1) * (section_size_x * RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))] - d_v_s[j + (e_ly - 1) * (section_size_x * RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))]);
                        p[nTid] = 0.0f;
                        //printf("div[tid] %f (%d, %d)| d_u[tid + 1] %f %f | d_u[tid - 1] %f %f| d_v[tid + N + 2] %f %f | d_v[tid - N - 2] %f %f\n",
                        //    div[nTid], ix, iy, d_u_s[(e_lx + 1) + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))], d_u[nTid + 1], d_u_s[(e_lx - 1) + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))], d_u[nTid - 1], 
                        //    d_v_s[j + (e_ly + 1) * (section_size_x * RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))], d_v[nTid + N + 2], d_v_s[j + (e_ly - 1) * (section_size_x * RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))], d_v[nTid - N - 2]);
                    }
                }
            }
        }
    }    

    // Handle remaining elements
    for (i = UNROLL_FACTOR; i < section_size_y && start_y + i < size; i++) {
        for (j = UNROLL_FACTOR; j < section_size_x && start_x + j < size; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);

            e_lx = j + RADIUS;
            e_ly = i + RADIUS;

            if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) {
                div[nTid] = -0.5f * h * (d_u_s[(e_lx + 1) + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))] - d_u_s[(e_lx - 1) + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))]
                    + d_v_s[j + (e_ly + 1) * (section_size_x * RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))] - d_v_s[j + (e_ly - 1) * (section_size_x * RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))]);
                p[nTid] = 0.0f;
                //printf("div[tid] %f (%d, %d)| d_u[tid + 1] %f %f | d_u[tid - 1] %f %f| d_v[tid + N + 2] %f %f | d_v[tid - N - 2] %f %f\n",
                //    div[nTid], ix, iy, d_u_s[(e_lx + 1) + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))], d_u[nTid + 1], d_u_s[(e_lx - 1) + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y))], d_u[nTid - 1], 
                //    d_v_s[j + (e_ly + 1) * (section_size_x * RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))], d_v[nTid + N + 2], d_v_s[j + (e_ly - 1) * (section_size_x * RADIUS) + lTid * ((section_size_x) * (section_size_y + 2))], d_v[nTid - N - 2]);
            }
        }
    }

    // maybe this is less performant ...
    setBordersOnGPU(0, div, ix, iy, start_x, start_y, section_size_x, section_size_y);
    setBordersOnGPU(0, p, ix, iy, start_x, start_y, section_size_x, section_size_y);
}

template <int UNROLL_FACTOR>
// CUDA kernel to perform the LAST PROJECTION STEP (using Jacobi iteration outside of kernel)
__global__ void lastProjectOnGPU(float *d_u, float *d_v, float *p) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int lTid = threadIdx.x + threadIdx.y * blockDim.x;
    int i, j, nIX, nIY, nTid, e_lx, e_ly, sTid;

    int size = (N + 2);
    int section_size_x = (size - 1) / (blockDim.x * gridDim.x) + 1;
    int section_size_y = (size - 1) / (blockDim.y * gridDim.y) + 1;
    int sizeShared = section_size_x * section_size_y * blockDim.x * blockDim.y;

    int start_x = ix * section_size_x;
    int start_y = iy * section_size_y;

    extern __shared__ float sharedMem[];
    float *d_u_s = (float*)sharedMem;
    float *d_v_s = (float*)&sharedMem[sizeShared];
    float *p_s = (float*)&sharedMem[2 * sizeShared];

    float h = 1.0f / N;
    // Unrolled section processing
    if (start_y + (UNROLL_FACTOR - 1) < size) {
        #pragma unroll
        for (i = 0; i < UNROLL_FACTOR; i++) {
            if (start_x + (UNROLL_FACTOR - 1) < size) {
                #pragma unroll
                for (j = 0; j < UNROLL_FACTOR; j++) {
                    nIX = start_x + j;
                    nIY = start_y + i;
                    nTid = nIX + nIY * (N + 2);
                    sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

                    e_lx = j + RADIUS;
                    e_ly = i + RADIUS;

                    if (nIX < (N + 2) && nIY < (N + 2)) {
                        if (i < RADIUS) { // copy top and bottom halo
                            //Copy Top Halo Element
		                    if (iy > 0) // Boundary Check
                                p_s[e_lx + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = p[nTid - RADIUS * (N + 2)];

		                    //Copy Bottom Halo Element
		                    if (iy < blockDim.y * gridDim.y - 1) // Boundary Check
                                p_s[e_lx + (e_ly + section_size_y) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = p[nTid + section_size_y * (N + 2)];
                        }  

                        if (j < RADIUS) { // copy left and right halo
                            // Copy Left Halo Element
                            if (ix > 0) // Boundary Check
                                p_s[j + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = p[nTid - RADIUS];
        
                            // Copy Right Halo Element
                            if (ix < blockDim.x * gridDim.x - 1) // Boundary Check
                                p_s[(e_lx + section_size_x) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = p[nTid + section_size_x];
                        }
                    

                        p_s[e_lx + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = p[nTid];
                        d_u_s[sTid] = d_u[nTid];
                        d_v_s[sTid] = d_v[nTid];
                        //printf("OLD nTid %d (%d %d) sTid %d lTid %d | d_u %f %f | d_v %f %f | p 1 %f %f | p 2 %f %f | p 3 %f %f | p 4 %f %f\n",
                        //    nTid, nIX, nIY, sTid, lTid, d_u_s[sTid], d_u[nTid], d_v_s[sTid], d_v[nTid], p_s[(e_lx + 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid + 1],
                        //    p_s[(e_lx - 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid - 1], p_s[e_lx + (e_ly + 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid + N + 2],
                        //    p_s[e_lx + (e_ly - 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid - N - 2]);
                    }
                }
            }
        }
    }

    // Handle remaining elements
    for (i = UNROLL_FACTOR; i < section_size_y && start_y + i < size; i++) {
        for (j = UNROLL_FACTOR; j < section_size_x && start_x + j < size; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);
            sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

            e_lx = j + RADIUS;
            e_ly = i + RADIUS;

            if (nIX < (N + 2) && nIY < (N + 2)) {
                if (i < RADIUS) { // copy top and bottom halo
                    //Copy Top Halo Element
		            if (iy > 0) // Boundary Check
                        p_s[e_lx + i * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = p[nTid - RADIUS * (N + 2)];

		            //Copy Bottom Halo Element
		            if (iy < blockDim.y * gridDim.y - 1) // Boundary Check
                        p_s[e_lx + (e_ly + section_size_y) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = p[nTid + section_size_y * (N + 2)];
                    }  

                if (j < RADIUS) { // copy left and right halo
                    // Copy Left Halo Element
                    if (ix > 0) // Boundary Check
                        p_s[j + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = p[nTid - RADIUS];
        
                    // Copy Right Halo Element
                    if (ix < blockDim.x * gridDim.x - 1) // Boundary Check
                        p_s[(e_lx + section_size_x) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = p[nTid + section_size_x];
                }

                p_s[e_lx + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] = p[nTid];
                d_u_s[sTid] = d_u[nTid];
                d_v_s[sTid] = d_v[nTid];
                //printf("OLD nTid %d (%d %d) sTid %d lTid %d | d_u %f %f | d_v %f %f | p 1 %f %f | p 2 %f %f | p 3 %f %f | p 4 %f %f\n",
                //    nTid, nIX, nIY, sTid, lTid, d_u_s[sTid], d_u[nTid], d_v_s[sTid], d_v[nTid], p_s[(e_lx + 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid + 1],
                //    p_s[(e_lx - 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid - 1], p_s[e_lx + (e_ly + 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid + N + 2],
                //    p_s[e_lx + (e_ly - 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid - N - 2]);
            }
        }
    }

    __syncthreads();

    // Unrolled section processing
    if (start_y + (UNROLL_FACTOR - 1) < size) {
        #pragma unroll
        for (i = 0; i < UNROLL_FACTOR; i++) {
            if (start_x + (UNROLL_FACTOR - 1) < size) {
                #pragma unroll
                for (j = 0; j < UNROLL_FACTOR; j++) {
                    nIX = start_x + j;
                    nIY = start_y + i;
                    nTid = nIX + nIY * (N + 2);
                    sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

                    e_lx = j + RADIUS;
                    e_ly = i + RADIUS;

                    if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) {
                        d_u[nTid] = d_u_s[sTid] - 0.5f * (p_s[(e_lx + 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] - p_s[(e_lx - 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))]) / h;
                        d_v[nTid] = d_v_s[sTid] - 0.5f * (p_s[e_lx + (e_ly + 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] - p_s[e_lx + (e_ly - 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))]) / h;
                        //printf("nTid %d (%d %d) sTid %d lTid %d | d_u %f %f | d_v %f %f | p 1 %f %f | p 2 %f %f | p 3 %f %f | p 4 %f %f\n",
                        //    nTid, nIX, nIY, sTid, lTid, d_u_s[sTid], d_u[nTid], d_v_s[sTid], d_v[nTid], p_s[(e_lx + 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid + 1],
                        //    p_s[(e_lx - 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid - 1], p_s[e_lx + (e_ly + 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid + N + 2],
                        //    p_s[e_lx + (e_ly - 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid - N - 2]);
                    }
                }
            }
        }
    }

    // Handle remaining elements
    for (i = UNROLL_FACTOR; i < section_size_y && start_y + i < size; i++) {
        for (j = UNROLL_FACTOR; j < section_size_x && start_x + j < size; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);
            sTid = (j + i * section_size_x) + lTid * (section_size_x * section_size_y);

            e_lx = j + RADIUS;
            e_ly = i + RADIUS;

            if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) {
                d_u[nTid] = d_u_s[sTid] - 0.5f * (p_s[(e_lx + 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] - p_s[(e_lx - 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))]) / h;
                d_v[nTid] = d_v_s[sTid] - 0.5f * (p_s[e_lx + (e_ly + 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))] - p_s[e_lx + (e_ly - 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))]) / h;
                //printf("nTid %d (%d %d) sTid %d lTid %d | d_u %f %f | d_v %f %f | p 1 %f %f | p 2 %f %f | p 3 %f %f | p 4 %f %f\n",
                //    nTid, nIX, nIY, sTid, lTid, d_u_s[sTid], d_u[nTid], d_v_s[sTid], d_v[nTid], p_s[(e_lx + 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid + 1],
                //    p_s[(e_lx - 1) + e_ly * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid - 1], p_s[e_lx + (e_ly + 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid + N + 2],
                //    p_s[e_lx + (e_ly - 1) * (section_size_x + 2 * RADIUS) + lTid * ((section_size_x + 2) * (section_size_y + 2))], p[nTid - N - 2]);
            }
        }
    }

    // can be done better but it works
    setBordersOnGPU(1, d_u, ix, iy, start_x, start_y, section_size_x, section_size_y);
    setBordersOnGPU(2, d_v, ix, iy, start_x, start_y, section_size_x, section_size_y);
    
    /*
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int i, j, nIX, nIY, nTid;

    int size = (N + 2);
    int section_size_x = (size - 1) / (blockDim.x * gridDim.x) + 1;
    int section_size_y = (size - 1) / (blockDim.y * gridDim.y) + 1;

    int start_x = ix * section_size_x;
    int start_y = iy * section_size_y;

    float h = 1.0f / N;
    // Unrolled section processing
    if (start_y + (UNROLL_FACTOR - 1) < size) {
        #pragma unroll
        for (i = 0; i < UNROLL_FACTOR; i++) {
            if (start_x + (UNROLL_FACTOR - 1) < size) {
                #pragma unroll
                for (j = 0; j < UNROLL_FACTOR; j++) {
                    nIX = start_x + j;
                    nIY = start_y + i;
                    nTid = nIX + nIY * (N + 2);
                    if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) {
                        d_u[nTid] -= 0.5f * (p[nTid + 1] - p[nTid - 1]) / h;
                        d_v[nTid] -= 0.5f * (p[nTid + N + 2] - p[nTid - N - 2]) / h;
                    }
                }
            }
        }
    }

    // Handle remaining elements
    for (i = UNROLL_FACTOR; i < section_size_y && start_y + i < size; i++) {
        for (j = UNROLL_FACTOR; j < section_size_x && start_x + j < size; j++) {
            nIX = start_x + j;
            nIY = start_y + i;
            nTid = nIX + nIY * (N + 2);
            if (nIX >= 1 && nIX <= N && nIY >= 1 && nIY <= N) {
                d_u[nTid] -= 0.5f * (p[nTid + 1] - p[nTid - 1]) / h;
                d_v[nTid] -= 0.5f * (p[nTid + N + 2] - p[nTid - N - 2]) / h;
            }
        }
    }

    // can be done better but it works
    setBordersOnGPU(1, d_u, ix, iy, start_x, start_y, section_size_x, section_size_y);
    setBordersOnGPU(2, d_v, ix, iy, start_x, start_y, section_size_x, section_size_y);
    */
}

// Function to simulate the evolution of density
void dens_step(dim3 grid, dim3 block, float *d_x, float *d_x0, float *d_u, float *d_v, float *d_densTemp) {
    int size = (hN + 2);
    int section_size_x = (size - 1) / (block.x * grid.x) + 1;
    int section_size_y = (size - 1) / (block.y * grid.y) + 1;

    size_t sizeShared = section_size_x * section_size_y * block.x * block.y * sizeof(float);
    size_t sizeSharedMax = (section_size_x + 2 * hRADIUS) * (section_size_y + 2 * hRADIUS) * block.x * block.y * sizeof(float);

    add_sourceOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block, 2 * sizeShared>>>(d_x, d_x0);

    float alpha = hDT * hDIFF * hN * hN;
    float beta = 1 + 4 * alpha;
    SWAP(d_x0, d_x);
    for (int k = 0; k < 40; k++) { // inefficient -> multiple kernel calls
        diffuseOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block, sizeShared + sizeSharedMax>>>(0, d_x, d_x0, d_densTemp, alpha, beta);
        SWAP(d_densTemp, d_x);
    }
    
    SWAP(d_x0, d_x);
    advectOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block, 4 * sizeShared>>>(0, d_x, d_x0, d_u, d_v);
}

// Function to simulate the evolution of velocity
void vel_step(dim3 grid, dim3 block, float *d_u, float *d_v, float *d_u0, float *d_v0, float *d_uTemp, float *d_vTemp) {
    int size = (hN + 2);
    int section_size_x = (size - 1) / (block.x * grid.x) + 1;
    int section_size_y = (size - 1) / (block.y * grid.y) + 1;

    size_t sizeShared = section_size_x * section_size_y * block.x * block.y * sizeof(float);
    size_t sizeSharedMax = (section_size_x + 2 * hRADIUS) * (section_size_y + 2 * hRADIUS) * block.x * block.y * sizeof(float);
    size_t sizeSharedU = (section_size_x + 2 * hRADIUS) * section_size_y * block.x * block.y * sizeof(float);
    size_t sizeSharedV = section_size_x * (section_size_y + 2 * hRADIUS) * block.x * block.y * sizeof(float);

    add_sourceOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block, 2 * sizeShared>>>(d_u, d_u0);
    add_sourceOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block, 2 * sizeShared>>>(d_v, d_v0);

    SWAP(d_u, d_u0);
    SWAP(d_v, d_v0);
    /*
    cudaDeviceSynchronize();
    float *temp = (float*)malloc(sizeof(float) * (hN + 2) * (hN + 2));
    cudaMemcpy(temp, d_u0, sizeof(float) * (hN + 2) * (hN + 2), cudaMemcpyDeviceToHost);
    printf("AFTER SOURCE u\n");
    printDebug(temp);
    */

    float alpha = hDT * hVIS * hN * hN;
    float beta = 1 + 4 * alpha;
    for (int k = 0; k < 40; k++) { // inefficient -> multiple kernel calls
        diffuseOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block, sizeShared + sizeSharedMax>>>(1, d_u, d_u0, d_uTemp, alpha, beta);
        diffuseOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block, sizeShared + sizeSharedMax>>>(2, d_v, d_v0, d_vTemp, alpha, beta);
        SWAP(d_uTemp, d_u);
        SWAP(d_vTemp, d_v);
    }
    computeDivergenceAndPressureOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block, sizeSharedU + sizeSharedV>>>(d_u, d_v, d_u0, d_v0);

    alpha = 1;
    beta = 4;
    // d_u0 is p, d_v0 is div
    for (int k = 0; k < 40; k++) { // inefficient -> multiple kernel calls
        diffuseOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block, sizeShared + sizeSharedMax>>>(0, d_u0, d_v0, d_uTemp, alpha, beta);
        SWAP(d_uTemp, d_u0);
    }
    lastProjectOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block, 2 * sizeShared + sizeSharedMax>>>(d_u, d_v, d_u0);

    SWAP(d_u0, d_u);
    SWAP(d_v0, d_v);
    advectOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block, 4 * sizeShared>>>(1, d_u, d_u0, d_u0, d_v0); 
    advectOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block, 4 * sizeShared>>>(2, d_v, d_v0, d_u0, d_v0);

    computeDivergenceAndPressureOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block,  sizeSharedU + sizeSharedV>>>(d_u, d_v, d_u0, d_v0);
    // d_u0 is p, d_v0 is div
    for (int k = 0; k < 40; k++) { // inefficient -> multiple kernel calls
        diffuseOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block, sizeShared + sizeSharedMax>>>(0, d_u0, d_v0, d_uTemp, alpha, beta);
        SWAP(d_uTemp, d_u0);
    }
    lastProjectOnGPU<1 << GRID_DIVISION_FACTOR><<<grid, block, 2 * sizeShared + sizeSharedMax>>>(d_u, d_v, d_u0);
    
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
    CHECK(cudaMemcpyToSymbol(RADIUS, &hRADIUS, sizeof(int)));

    // Define grid and block dimensions
    dim3 block, grid, nGrid;
    block = dim3(block_dim_x, block_dim_y);
    grid = dim3(((hN + 2) + block_dim_x - 1) / block_dim_x, ((hN + 2) + block_dim_y - 1) / block_dim_y);
    nGrid = dim3(grid.x / (1 << GRID_DIVISION_FACTOR), grid.y / (1 << GRID_DIVISION_FACTOR));

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

        vel_step(nGrid, block, d_u, d_v, d_u_prev, d_v_prev, d_uTemp, d_vTemp);
        dens_step(nGrid, block, d_dens, d_dens_prev, d_u, d_v, d_densTemp);

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
    printf("grid: %d, <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", size, nGrid.x, nGrid.y, block.x, block.y, iElaps);

    CHECK(cudaMemcpy(u, d_u, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(v, d_v, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(dens, d_dens, nBytes, cudaMemcpyDeviceToHost));
    //printStateGrid(dens, u, v);        

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

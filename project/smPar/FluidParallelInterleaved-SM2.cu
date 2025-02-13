// CUDA first version naive (multiple cells for each thread (strided))
// Using Shared Memory

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
int hN = (1<<13) - 2;
float hDT = 0.016f;
float hVIS = 0.0025f;
float hDIFF = 0.1f;
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

/*
__global__ void boundaryGPU(int b, float *d_x) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (blockIdx.x == 0) {
        set_bndOnGPU(b, d_x, ix, 0, ix);
        set_bndOnGPU(b, d_x, ix, N + 1, ix + (N + 1) * (N + 2));
        set_bndOnGPU(b, d_x, 0, ix, ix * (N + 2));
        set_bndOnGPU(b, d_x, 0, N + 1 - ix, (N + 1 - ix) * (N + 2));
        __syncthreads();
        set_crnOnGPU(b, d_x, 0, 0, 0);
        set_crnOnGPU(b, d_x, 0, N + 1, (N + 1) * (N + 2));
    } else if (blockIdx.x > 0 && blockIdx.x < gridDim.x - 1) {
        set_bndOnGPU(b, d_x, ix, 0, ix);
        set_bndOnGPU(b, d_x, 0, ix, ix * (N + 2));
        set_bndOnGPU(b, d_x, N + 1, ix, N + 1 + ix * (N + 2));
        set_bndOnGPU(b, d_x, ix, N + 1, ix + (N + 1) * (N + 2));
    } else if (blockIdx.x == gridDim.x - 1) {
        set_bndOnGPU(b, d_x, ix, 0, ix);
        set_bndOnGPU(b, d_x, ix, N + 1, ix + (N + 1) * (N + 2));
        set_bndOnGPU(b, d_x, N + 1, ix, N + 1 + ix * (N + 2));
        set_bndOnGPU(b, d_x, N + 1, N + 1 - ix, N + 1 + (N + 1 - ix) * (N + 2));
        __syncthreads();
        set_crnOnGPU(b, d_x, N + 1, 0, N + 1);
        set_crnOnGPU(b, d_x, N + 1, N + 1, (N + 1) + (N + 1) * (N + 2));
    }
}
*/

// CUDA kernel function to  ADD EXTERNAL SOURCES
__global__ void add_sourceOnGPU(float *d_x, float *d_s) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int lTid = threadIdx.x + threadIdx.y * blockDim.x;
    int i, j, k, n, tid, sTid;

    int size = (N + 2);
    int section_size_x = (size - 1) / (blockDim.x * gridDim.x) + 1;
    int section_size_y = (size - 1) / (blockDim.y * gridDim.y) + 1;
    int sizeShared = section_size_x * section_size_y * blockDim.x * blockDim.y;

    extern __shared__ float sharedMem[];
    float *d_x_s = (float*)sharedMem;
    float *d_s_s = (float*)&sharedMem[sizeShared];

    for (i = iy, k = 0; i < size; i += blockDim.y * gridDim.y, k++) {
        for (j = ix, n = 0; j < size; j += blockDim.x * gridDim.x, n++) {
            tid = j + i * (N + 2);
            sTid = (n + k * section_size_x) + lTid * (section_size_x * section_size_y);

            if (j < (N + 2) && i < (N + 2)) {
                d_x_s[sTid] = d_x[tid];
                d_s_s[sTid] = d_s[tid];
            }
        }
    }

    __syncthreads();

    for (i = iy, k = 0; i < size; i += blockDim.y * gridDim.y, k++) {
        for (j = ix, n = 0; j < size; j += blockDim.x * gridDim.x, n++) {
            tid = j + i * (N + 2);
            sTid = (n + k * section_size_x) + lTid * (section_size_x * section_size_y);

            if (j < (N + 2) && i < (N + 2))
                d_x[tid] = d_x_s[sTid] + DT * d_s_s[sTid];
        }
    }
}

// CUDA kernel function to perform DIFFUSION (using Jacobi iteration outside of kernel)
__global__ void diffuseOnGPU(int b, float *d_x, float *d_x0, float *d_xTemp, float alpha, float beta) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int lTid = threadIdx.x + threadIdx.y * blockDim.x;
    int i, j, k, n, tid, sTid;

    int size = (N + 2);
    int section_size_x = (size - 1) / (blockDim.x * gridDim.x) + 1;
    int section_size_y = (size - 1) / (blockDim.y * gridDim.y) + 1;
    int sizeShared = section_size_x * section_size_y * blockDim.x * blockDim.y;

    int start_x = ix * section_size_x;
    int start_y = iy * section_size_y;

    extern __shared__ float sharedMem[];
    float *d_x0_s = (float*)sharedMem;
    float *d_x_s = (float*)&sharedMem[sizeShared];


    for (i = iy, k = 0; i < size; i += blockDim.y * gridDim.y, k++) {
        for (j = ix, n = 0; j < size; j += blockDim.x * gridDim.x, n++) {
            tid = j + i * (N + 2);
            sTid = (n + k * section_size_x) + lTid * (section_size_x * section_size_y);

            if (j >= 1 && j <= N && i >= 1 && i <= N) {
                d_x_s[sTid] = d_x[tid - 1];
                d_x_s[sTid + sizeShared] = d_x[tid + 1];
                d_x_s[sTid + 2 * sizeShared] = d_x[tid - N - 2];
                d_x_s[sTid + 3 * sizeShared] = d_x[tid + N + 2];
            }

            if (j < N + 2 && i < N + 2)
                d_x0_s[sTid] = d_x0[tid];

            //if (j >= 1 && j <= N && i >= 1 && i <= N)
            //printf("tid %d (%d %d) sTid %d | d_x0 %f %f | d_x 1 %f %f | d_x 2 %f %f | d_x 3 %f %f | d_x 4 %f %f\n",
            //    tid, j, i, sTid, d_x0_s[sTid], d_x0[tid], d_x_s[sTid], d_x[tid - 1], d_x_s[sTid + sizeShared], d_x[tid + 1], d_x_s[sTid + 2 * sizeShared], d_x[tid - N - 2],
            //    d_x_s[sTid + 3 * sizeShared], d_x[tid + N + 2]);
        }
    }

    __syncthreads();


    for (i = iy, k = 0; i < size; i += blockDim.y * gridDim.y, k++) {
        for (j = ix, n = 0; j < size; j += blockDim.x * gridDim.x, n++) {
            tid = j + i * (N + 2);
            sTid = (n + k * section_size_x) + lTid * (section_size_x * section_size_y);

            if (j >= 1 && j <= N && i >= 1 && i <= N) {
                d_xTemp[tid] = (d_x0_s[sTid] + alpha * (d_x_s[sTid] + d_x_s[sTid + sizeShared] + 
                    d_x_s[sTid + 2 * sizeShared] + d_x_s[sTid + 3 * sizeShared])) / beta;
                //printf("tid %d (%d %d) sTid %d | d_x0 %f | d_x 1 %f %f | d_x 2 %f %f | d_x 3 %f %f | d_x 4 %f %f\n",
                //    tid, j, i, sTid, d_x0_s[sTid], d_x_s[sTid], d_x[tid - 1], d_x_s[sTid + sizeShared], d_x[tid + 1], d_x_s[sTid + 2 * sizeShared], d_x[tid - N - 2],
                //    d_x_s[sTid + 3 * sizeShared], d_x[tid + N + 2]);
            }
        }
    }

    setBordersOnGPU(b, d_xTemp, ix, iy, start_x, start_y, section_size_x, section_size_y);
}

// CUDA kernel function to perform ADVECTION (using bilinear interpolation)
__global__ void advectOnGPU(int b, float *d_d, float *d_d0, float *d_u, float *d_v) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int lTid = threadIdx.x + threadIdx.y * blockDim.x;
    int i, j, k, n, tid, sTid;

    int size = (N + 2);
    int section_size_x = (size - 1) / (blockDim.x * gridDim.x) + 1;
    int section_size_y = (size - 1) / (blockDim.y * gridDim.y) + 1;
    int sizeShared = section_size_x * section_size_y * blockDim.x * blockDim.y;

    int start_x = ix * section_size_x;
    int start_y = iy * section_size_y;

    int i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    extern __shared__ float sharedMem[];
    float *d_d0_s = (float*)sharedMem;

    dt0 = DT * N;
    for (i = iy, k = 0; i < size; i += blockDim.y * gridDim.y, k++) {
        for (j = ix, n = 0; j < size; j += blockDim.x * gridDim.x, n++) {
            tid = j + i * (N + 2);
            sTid = (n + k * section_size_x) + lTid * (section_size_x * section_size_y);

            if (j >= 1 && j <= N && i >= 1 && i <= N) {
                x = j - dt0 * d_u[tid];
                y = i - dt0 * d_v[tid];

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

 
    for (i = iy, k = 0; i < size; i += blockDim.y * gridDim.y, k++) {
        for (j = ix, n = 0; j < size; j += blockDim.x * gridDim.x, n++) {
            tid = j + i * (N + 2);
            sTid = (n + k * section_size_x) + lTid * (section_size_x * section_size_y);

            if (j >= 1 && j <= N && i >= 1 && i <= N) {
                d_d[tid] = d_d0_s[sTid] + d_d0_s[sTid + sizeShared] + d_d0_s[sTid + 2 * sizeShared] + d_d0_s[sTid + 3 * sizeShared];
                //printf("(%f, %f) | d_d[tid] %f (%d, %d) | d_d0[j0 + i0 * (N + 2)] %f | d_d0[j0 + i1 * (N + 2)] %f | d_d0[j1 + i0 * (N + 2)] %f | d_d0[j1 + i1 * (N + 2)] %f\n",
                //    x, y, d_d[tid], ix, iy, d_d0[j0 + i0 * (N + 2)], d_d0[j0 + i1 * (N + 2)], d_d0[j1 + i0 * (N + 2)], d_d0[j1 + i1 * (N + 2)]);
            }
        }
    }

    setBordersOnGPU(b, d_d, ix, iy, start_x, start_y, section_size_x, section_size_y);
}

// CUDA kernel function to COMPUTE DIVERGENCE AND PRESSURE
__global__ void computeDivergenceAndPressureOnGPU(float *d_u, float *d_v, float *p, float *div) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int lTid = threadIdx.x + threadIdx.y * blockDim.x;
    int i, j, k, n, tid, sTid;

    int size = (N + 2);
    int section_size_x = (size - 1) / (blockDim.x * gridDim.x) + 1;
    int section_size_y = (size - 1) / (blockDim.y * gridDim.y) + 1;
    int sizeShared = section_size_x * section_size_y * blockDim.x * blockDim.y;

    int start_x = ix * section_size_x;
    int start_y = iy * section_size_y;

    extern __shared__ float sharedMem[];
    float *d_u_s = (float*)sharedMem;
    float *d_v_s = (float*)&sharedMem[2 * sizeShared];

    float h = 1.0f / N;
    for (i = iy, k = 0; i < size; i += blockDim.y * gridDim.y, k++) {
        for (j = ix, n = 0; j < size; j += blockDim.x * gridDim.x, n++) {
            tid = j + i * (N + 2);
            sTid = (n + k * section_size_x) + lTid * (section_size_x * section_size_y);

            if (j >= 1 && j <= N && i >= 1 && i <= N) {
                d_v_s[sTid] = d_v[tid - N - 2];
                d_v_s[sTid + sizeShared] = d_v[tid + N + 2];
                d_u_s[sTid] = d_u[tid -1];
                d_u_s[sTid + sizeShared] = d_u[tid + 1];
            }
        }
    }

    __syncthreads();
    /*
    for (i = iy, k = 0; i < size; i += blockDim.y * gridDim.y, k++) {
        for (j = ix, n = 0; j < size; j += blockDim.x * gridDim.x, n++) {
            tid = j + i * (N + 2);
            sTid = (n + k * section_size_x) + lTid * (section_size_x * section_size_y);

            if (j >= 1 && j <= N && i >= 1 && i <= N) {
                printf("sTid %d tid %d (%d %d) | d_v 1 %f %f | d_v 2 %f %f | d_u 1 %f %f | d_u 2 %f %f\n",
                    sTid, tid, j, i, d_v_s[sTid], d_v[tid - N - 2], d_v_s[sTid + sizeShared], d_v[tid + N + 2],
                    d_u_s[sTid], d_u[tid -1], d_u_s[sTid + sizeShared], d_u[tid + 1]);
            }
        }
    }
    */

    for (i = iy, k = 0; i < size; i += blockDim.y * gridDim.y, k++) {
        for (j = ix, n = 0; j < size; j += blockDim.x * gridDim.x, n++) {
            tid = j + i * (N + 2);
            sTid = (n + k * section_size_x) + lTid * (section_size_x * section_size_y);

            if (j >= 1 && j <= N && i >= 1 && i <= N) {
                div[tid] = -0.5f * h * (d_u_s[sTid + sizeShared] - d_u_s[sTid] + d_v_s[sTid + sizeShared] - d_v_s[sTid]);
                p[tid] = 0.0f;
                //printf("div[tid] %f (%d, %d)| d_u[tid + 1] %f | d_u[tid - 1] %f | d_v[tid + N + 2] %f | d_v[tid - N - 2] %f\n",
                //    div[tid], ix, iy, d_u[tid + 1], d_u[tid - 1], d_v[tid + N + 2], d_v[tid - N - 2]);
            }
        }
    }

    setBordersOnGPU(0, div, ix, iy, start_x, start_y, section_size_x, section_size_y);
    setBordersOnGPU(0, p, ix, iy, start_x, start_y, section_size_x, section_size_y);
}

// CUDA kernel to perform the LAST PROJECTION STEP (using Jacobi iteration outside of kernel)
__global__ void lastProjectOnGPU(float *d_u, float *d_v, float *p) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int lTid = threadIdx.x + threadIdx.y * blockDim.x;
    int i, j, k, n, tid, sTid;

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
    for (i = iy, k = 0; i < size; i += blockDim.y * gridDim.y, k++) {
        for (j = ix, n = 0; j < size; j += blockDim.x * gridDim.x, n++) {
            tid = j + i * (N + 2);
            sTid = (n + k * section_size_x) + lTid * (section_size_x * section_size_y);

            if (j >= 1 && j <= N && i >= 1 && i <= N) {
                p_s[sTid] = p[tid - 1];
                p_s[sTid + sizeShared] = p[tid + 1];
                p_s[sTid + 2 * sizeShared] = p[tid - N - 2];
                p_s[sTid + 3 * sizeShared] = p[tid + N + 2];
            }

            if (j < N + 2 && i < N + 2) {
                d_u_s[sTid] = d_u[tid];
                d_v_s[sTid] = d_v[tid];
            }
        }
    }

    __syncthreads();

    for (i = iy, k = 0; i < size; i += blockDim.y * gridDim.y, k++) {
        for (j = ix, n = 0; j < size; j += blockDim.x * gridDim.x, n++) {
            tid = j + i * (N + 2);
            sTid = (n + k * section_size_x) + lTid * (section_size_x * section_size_y);

            if (j >= 1 && j <= N && i >= 1 && i <= N) {
                d_u[tid] = d_u_s[sTid] - 0.5f * (p_s[sTid + sizeShared] - p_s[sTid]) / h;
                d_v[tid] = d_v_s[sTid] - 0.5f * (p_s[sTid + 3 * sizeShared] - p_s[sTid + 2 * sizeShared]) / h;
            }
        }
    }

    setBordersOnGPU(1, d_u, ix, iy, start_x, start_y, section_size_x, section_size_y);
    setBordersOnGPU(2, d_v, ix, iy, start_x, start_y, section_size_x, section_size_y);
}

// Function to simulate the evolution of density
void dens_step(dim3 grid, dim3 block, int boundGrid, float *d_x, float *d_x0, float *d_u, float *d_v, float *d_densTemp) {
    int size = (hN + 2);
    int section_size_x = (size - 1) / (block.x * grid.x) + 1;
    int section_size_y = (size - 1) / (block.y * grid.y) + 1;

    size_t sizeShared = section_size_x * section_size_y * block.x * block.y * sizeof(float);
    
    add_sourceOnGPU<<<grid, block, 2 * sizeShared>>>(d_x, d_x0);

    float alpha = hDT * hDIFF * hN * hN;
    float beta = 1 + 4 * alpha;
    SWAP(d_x0, d_x);
    for (int k = 0; k < 40; k++) { // inefficient -> multiple kernel calls
        diffuseOnGPU<<<grid, block, 5 * sizeShared>>>(0, d_x, d_x0, d_densTemp, alpha, beta);
        //boundaryGPU<<<boundGrid, block.x>>>(0, d_densTemp);
        SWAP(d_densTemp, d_x);
    }
    
    SWAP(d_x0, d_x);
    advectOnGPU<<<grid, block, 4 * sizeShared>>>(0, d_x, d_x0, d_u, d_v);
    //boundaryGPU<<<boundGrid, block.x>>>(0, d_x);
}

// Function to simulate the evolution of velocity
void vel_step(dim3 grid, dim3 block, int boundGrid, float *d_u, float *d_v, float *d_u0, float *d_v0, float *d_uTemp, float *d_vTemp) {
    int size = (hN + 2);
    int section_size_x = (size - 1) / (block.x * grid.x) + 1;
    int section_size_y = (size - 1) / (block.y * grid.y) + 1;

    size_t sizeShared = section_size_x * section_size_y * block.x * block.y * sizeof(float);
    
    add_sourceOnGPU<<<grid, block, 2 * sizeShared>>>(d_u, d_u0);
    add_sourceOnGPU<<<grid, block, 2 * sizeShared>>>(d_v, d_v0);

    SWAP(d_u, d_u0);
    SWAP(d_v, d_v0);

    float alpha = hDT * hVIS * hN * hN;
    float beta = 1 + 4 * alpha;
    for (int k = 0; k < 40; k++) { // inefficient -> multiple kernel calls
        diffuseOnGPU<<<grid, block, 5 * sizeShared>>>(1, d_u, d_u0, d_uTemp, alpha, beta);
        diffuseOnGPU<<<grid, block, 5 * sizeShared>>>(2, d_v, d_v0, d_vTemp, alpha, beta);
        //boundaryGPU<<<boundGrid, block.x>>>(1, d_uTemp);
        //boundaryGPU<<<boundGrid, block.x>>>(2, d_vTemp);
        SWAP(d_uTemp, d_u);
        SWAP(d_vTemp, d_v);
    }
    
    computeDivergenceAndPressureOnGPU<<<grid, block, 4 * sizeShared>>>(d_u, d_v, d_u0, d_v0);
    //boundaryGPU<<<boundGrid, block.x>>>(0, d_u0);
    //boundaryGPU<<<boundGrid, block.x>>>(0, d_v0);

    alpha = 1;
    beta = 4;
    // d_u0 is p, d_v0 is div
    for (int k = 0; k < 40; k++) { // inefficient -> multiple kernel calls
        diffuseOnGPU<<<grid, block, 5 * sizeShared>>>(0, d_u0, d_v0, d_uTemp, alpha, beta);
        //boundaryGPU<<<boundGrid, block.x>>>(0, d_uTemp);
        SWAP(d_uTemp, d_u0);
    }
    lastProjectOnGPU<<<grid, block, 6 * sizeShared>>>(d_u, d_v, d_u0);
    //boundaryGPU<<<boundGrid, block.x>>>(1, d_u);
    //boundaryGPU<<<boundGrid, block.x>>>(2, d_v);

    SWAP(d_u0, d_u);
    SWAP(d_v0, d_v);
    advectOnGPU<<<grid, block, 4 * sizeShared>>>(1, d_u, d_u0, d_u0, d_v0);
    advectOnGPU<<<grid, block, 4 * sizeShared>>>(2, d_v, d_v0, d_u0, d_v0);
    //boundaryGPU<<<boundGrid, block.x>>>(1, d_u);
    //boundaryGPU<<<boundGrid, block.x>>>(2, d_v);

    computeDivergenceAndPressureOnGPU<<<grid, block, 4 * sizeShared>>>(d_u, d_v, d_u0, d_v0);
    //boundaryGPU<<<boundGrid, block.x>>>(0, d_u0);
    //boundaryGPU<<<boundGrid, block.x>>>(0, d_v0);
    // d_u0 is p, d_v0 is div
    for (int k = 0; k < 40; k++) { // inefficient -> multiple kernel calls
        diffuseOnGPU<<<grid, block, 5 * sizeShared>>>(0, d_u0, d_v0, d_uTemp, alpha, beta);
        //boundaryGPU<<<boundGrid, block.x>>>(0, d_uTemp);
        SWAP(d_uTemp, d_u0);
    }
    lastProjectOnGPU<<<grid, block, 6 * sizeShared>>>(d_u, d_v, d_u0);
    //boundaryGPU<<<boundGrid, block.x>>>(1, d_u);
    //boundaryGPU<<<boundGrid, block.x>>>(2, d_v);
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

        vel_step(nGrid, block, grid.x, d_u, d_v, d_u_prev, d_v_prev, d_uTemp, d_vTemp);
        dens_step(nGrid, block, grid.x, d_dens, d_dens_prev, d_u, d_v, d_densTemp);

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
// CUDA first version naive (1 cell for each thread)

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
        d_x[tid] = b == 2 ? -d_x[tid + blockDim.x] : d_x[tid + blockDim.x];
    else if (iy == N + 1 && ix > 0 && ix < N + 1)
        d_x[tid] = b == 2 ? -d_x[tid - blockDim.x] : d_x[tid - blockDim.x];
}   

// CUDA device function to set the boundary conditions (corners)
__device__ void set_crnOnGPU(int b, float *d_x, int ix, int iy, int tid) {
    if (ix == 0 && iy == 0)
        d_x[tid] = 0.5f * (d_x[tid + 1] + d_x[tid + blockDim.x]);
    else if (ix == 0 && iy == N + 1)
        d_x[tid] = 0.5f * (d_x[tid + 1] + d_x[tid - blockDim.x]);
    else if (ix == N + 1 && iy == 0)
        d_x[tid] = 0.5f * (d_x[tid - 1] + d_x[tid + blockDim.x]);
    else if (ix == N + 1 && iy == N + 1) 
        d_x[tid] = 0.5f * (d_x[tid - 1] + d_x[tid - blockDim.x]);
}


// CUDA kernel function to  ADD EXTERNAL SOURCES
__global__ void add_sourceOnGPU(float *d_x, float *d_s) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = ix + iy * (N + 2);
    int sizeShared = blockDim.x * blockDim.y;

    extern __shared__ float sharedMem[];
    float *d_x_s = (float*)sharedMem;
    float *d_s_s = (float*)&sharedMem[sizeShared];

    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int lTid = lx + ly * blockDim.x;

    //printf("lTid %d - (%d %d)\n", lTid, ix, iy);

    if (ix < (N + 2) && iy < (N + 2)) {
        d_x_s[lTid] = d_x[tid];
        d_s_s[lTid] = d_s[tid];
    }
    __syncthreads();
        //printf("sTid %d | tid %d | d_x_s[sTid] %f | d_s_s[sTid] %f, d_x[tid] %f | d_s[tid] %f\n",
        //    lTid, tid, d_x_s[lTid], d_s_s[lTid], d_x[tid], d_s[tid]);
    if (ix < (N + 2) && iy < (N + 2)) {
        d_x[tid] = d_x_s[lTid] + DT * d_s_s[lTid];
    }
    
}

// CUDA kernel function to perform DIFFUSION (using Jacobi iteration outside of kernel)
__global__ void diffuseOnGPU(int b, float *d_x, float *d_x0, float *d_xTemp, float alpha, float beta) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = ix + iy * (N + 2);
    int sizeShared = blockDim.x * blockDim.y;
    //int sizeSharedMax = (blockDim.x + 2 * RADIUS) * (blockDim.y + 2 * RADIUS);

    extern __shared__ float sharedMem[];
    float *d_x0_s = (float*)sharedMem;
    float *d_xTemp_s = (float*)&sharedMem[sizeShared];
    float *d_x_s = (float*)&sharedMem[2 * sizeShared];

    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int lTid = lx + ly * blockDim.x;

    // e_XX --> variables refers to expanded shared memory location in order to accomodate halo elements
	//Current Local ID with radius offset.
	int e_lx = lx + RADIUS;
	int e_ly = ly + RADIUS;

	// Variable pointing at top and bottom neighbouring location
	int e_ly_prev = e_ly - 1;
	int e_ly_next = e_ly + 1;

	// Variable pointing at left and right neighbouring location
	int e_lx_prev = e_lx - 1;
	int e_lx_next = e_lx + 1;

    if (ix < (N + 2) && iy < (N + 2)) {
        if (ly < RADIUS) { // copy top and bottom halo
            //Copy Top Halo Element
		    if (blockIdx.y > 0) // Boundary Check
                d_x_s[e_lx + ly * (blockDim.x + 2 * RADIUS)] = d_x[tid - RADIUS * (N + 2)];

		    //Copy Bottom Halo Element
		    if (blockIdx.y < (gridDim.y - 1)) // Boundary Check
                d_x_s[e_lx + (e_ly + blockDim.y) * (blockDim.x + 2 * RADIUS)] = d_x[tid + blockDim.y * (N + 2)];
        }  

        if (lx < RADIUS) { // copy left and right halo
            // Copy Left Halo Element
            if (blockIdx.x > 0) // Boundary Check
                d_x_s[lx + e_ly * (blockDim.x + 2 * RADIUS)] = d_x[tid - RADIUS];
        
            // Copy Right Halo Element
            if (blockIdx.x < (gridDim.x - 1)) // Boundary Check
                d_x_s[(e_lx + blockDim.x) + e_ly * (blockDim.x + 2 * RADIUS)] = d_x[tid + blockDim.x];
        }

        // Copy Current Location
        d_x_s[e_lx + e_ly * (blockDim.x + 2 * RADIUS)] = d_x[tid];
        d_x0_s[lTid] = d_x0[tid];
    }
    
    __syncthreads();

    if (ix >= 1 && ix <= N && iy >= 1 && iy <= N) {
        d_xTemp_s[lTid] = (d_x0_s[lTid] + alpha * (d_x_s[e_lx + e_ly_prev * (blockDim.x + 2 * RADIUS)] + d_x_s[e_lx + e_ly_next * (blockDim.x + 2 * RADIUS)] + 
            d_x_s[e_lx_prev + e_ly * (blockDim.x + 2 * RADIUS)] + d_x_s[e_lx_next + e_ly * (blockDim.x + 2 * RADIUS)])) / beta;
        //printf("d_xTemp_s id %f in (%d, %d) - lTid %d: d_x0[tid] %f | d_x[tid - 1] %f | d_x[tid + 1] %f, d_x[tid - N - 2] %f | d_x[tid + N + 2] %f | beta %f\n",
        //    d_xTemp_s[lTid], ix, iy, lTid, d_x0_s[lTid], d_x_s[e_lx_prev + e_ly * (blockDim.x + 2 * RADIUS)], d_x_s[e_lx_next + e_ly * (blockDim.x + 2 * RADIUS)], d_x_s[e_lx + e_ly_prev * (blockDim.x + 2 * RADIUS)], d_x_s[e_lx + e_ly_next * (blockDim.x + 2 * RADIUS)], beta);
    }
    
    // Inefficient because multiple synchronization
    if (blockIdx.x == 0 || blockIdx.x == gridDim.x - 1 || blockIdx.y == 0 || blockIdx.y == gridDim.y - 1 &&
        ix < N + 2 && iy < N + 2) {
        __syncthreads(); // synchronization intra-block to ensure that the 
                             // threads in the boundary can compute correctly
        set_bndOnGPU(b, d_xTemp_s, ix, iy, lTid);

        __syncthreads(); // now that all the borders are complete, we can compute
                          // corners
        set_crnOnGPU(b, d_xTemp_s, ix, iy, lTid);
    }
        
    if (ix < (N + 2) && iy < (N + 2)) {
        d_xTemp[tid] = d_xTemp_s[lTid];
    }
}

// CUDA kernel function to perform ADVECTION (using bilinear interpolation)
__global__ void advectOnGPU(int b, float *d_d, float *d_d0, float *d_u, float *d_v) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = ix + iy * (N + 2);
    int sizeShared = blockDim.x * blockDim.y;

    extern __shared__ float sharedMem[];
    float *d_u_s = (float*)sharedMem;
    float *d_v_s = (float*)&sharedMem[sizeShared];
    float *d_d_s = (float*)&sharedMem[2 * sizeShared];
    float *d_d0_s = (float*)&sharedMem[3 * sizeShared];

    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int lTid = lx + ly * blockDim.x;

    int i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    dt0 = DT * N;
    if (ix < (N + 2) && iy < (N + 2)) {
        d_u_s[lTid] = d_u[tid];
        d_v_s[lTid] = d_v[tid];
    }
    
    __syncthreads();
        //printf("FIRST lTid %d (%d, %d) | d_u_s[lTid] %f | d_u[tid] %f | d_v_s[lTid] %f | d_v[tid] %f\n",
        //    lTid, ix, iy, d_u_s[lTid], d_u[tid], d_v_s[lTid], d_v[tid]);
        
    if (ix >= 1 && ix <= N && iy >= 1 && iy <= N) {
        x = ix - dt0 * d_u_s[lTid];
        y = iy - dt0 * d_v_s[lTid];
        
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
        
        d_d0_s[lTid] = d_d0[j0 + i0 * (N + 2)];
        //__syncthreads();
        //printf("SECOND lTid %d (%d, %d) | d_d0_s[lTid] %f | d_d0[j0 + i0 * (N + 2)] %f\n",
        //    lTid, ix, iy, d_d0_s[lTid], d_d0[j0 + i0 * (N + 2)]);
        d_d0_s[lTid + sizeShared] = d_d0[j0 + i1 * (N + 2)];
        //__syncthreads();
        //printf("THIRD lTid %d (%d, %d) | d_d0_s[lTid + sizeShared] %f | d_d0[j0 + i1 * (N + 2)] %f\n",
        //    lTid, ix, iy, d_d0_s[lTid + sizeShared], d_d0[j0 + i1 * (N + 2)]);

        // PROBLEMA QUI?
        d_d0_s[lTid + 2 * sizeShared] = d_d0[j1 + i0 * (N + 2)];
        //__syncthreads();
        //printf("FOURTH lTid %d (%d, %d) | d_d0_s[lTid + 2 * sizeShared] %f | d_d0[j1 + i0 * (N + 2)] %f\n",
        //    lTid, ix, iy, d_d0_s[lTid + 2 * sizeShared], d_d0[j1 + i0 * (N + 2)]);

        
        d_d0_s[lTid + 3 * sizeShared] = d_d0[j1 + i1 * (N + 2)];
        //__syncthreads();
        //printf("FIFTH lTid %d (%d, %d) | d_d0_s[lTid + 3 * sizeShared] %f | d_d0[j0 + i0 * (N + 2)] %f\n",
        //    lTid, ix, iy, d_d0_s[lTid + 3 * sizeShared], d_d0[j1 + i1 * (N + 2)]);
    }
        
    __syncthreads();
        //printf("lTid %d (%d, %d) | d_d0_s[lTid] %f | d_d0[j0 + i0 * (N + 2)] %f | _d0_s[lTid + sizeShared] %f | d_d0[j0 + i1 * (N + 2)] %f | d_d0_s[lTid + 2 * sizeShared] %f | d_d0[j1 + i0 * (N + 2)] %f | d_d0_s[lTid + 3 * sizeShared] %f |d_d0[j1 + i1 * (N + 2)] %f\n",
        //    lTid, ix, iy, d_d0_s[lTid], d_d0[j0 + i0 * (N + 2)], d_d0_s[lTid + sizeShared], d_d0[j0 + i1 * (N + 2)], d_d0_s[lTid + 2 * sizeShared], d_d0[j1 + i0 * (N + 2)], d_d0_s[lTid + 3 * sizeShared], d_d0[j1 + i1 * (N + 2)]);
        
    if (ix >= 1 && ix <= N && iy >= 1 && iy <= N) {
        d_d_s[lTid] = s0 * (t0 * d_d0_s[lTid] + t1 * d_d0_s[lTid + sizeShared]) +
            s1 * (t0 * d_d0_s[lTid + 2 * sizeShared] + t1 * d_d0_s[lTid + 3 * sizeShared]);
        //printf("(%f, %f) | d_d0[tid] %f (%d, %d) - lTid %d | d_d0[j0 + i0 * (N + 2)] %f | d_d0[j0 + i1 * (N + 2)] %f | d_d0[j1 + i0 * (N + 2)] %f | d_d0[j1 + i1 * (N + 2)] %f\n",
        //    x, y, d_d0_s[lTid], ix, iy, lTid, d_d0_s[lTid + sizeShared], d_d0_s[lTid + 2 * sizeShared], d_d0_s[lTid + 2 * sizeShared], d_d0_s[lTid + 3 * sizeShared]);
    } 
    
    // Inefficient because multiple synchronization
    if (blockIdx.x == 0 || blockIdx.x == gridDim.x - 1 || blockIdx.y == 0 || blockIdx.y == gridDim.y - 1 &&
        ix < N + 2 && iy < N + 2) {
        __syncthreads(); // synchronization intra-block to ensure that the 
                         // threads in the boundary can compute correctly
        set_bndOnGPU(b, d_d_s, ix, iy, lTid);

        __syncthreads(); // now that all the borders are complete, we can compute
                          // corners
        set_crnOnGPU(b, d_d_s, ix, iy, lTid);
    }

    if (ix < (N + 2) && iy < (N + 2)) {
        d_d[tid] = d_d_s[lTid];
    }    
}

// CUDA kernel function to COMPUTE DIVERGENCE AND PRESSURE
__global__ void computeDivergenceAndPressureOnGPU(float *d_u, float *d_v, float *p, float *div) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = ix + iy * (N + 2);
    int sizeShared = blockDim.x * blockDim.y;
    int sizeSharedU = (blockDim.x + 2) * blockDim.y;
    int sizeSharedV = blockDim.x * (blockDim.y + 2);

    extern __shared__ float sharedMem[];
    float *d_u_s = (float*)sharedMem;
    float *d_v_s = (float*)&sharedMem[sizeSharedU];
    float *div_s = (float*)&sharedMem[sizeSharedU + sizeSharedV];
    float *p_s = (float*)&sharedMem[sizeSharedU + sizeSharedV + sizeShared];

    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int lTid = lx + ly * blockDim.x;

    // e_XX --> variables refers to expanded shared memory location in order to accomodate halo elements
	//Current Local ID with radius offset.
	int e_lx = lx + RADIUS;
	int e_ly = ly + RADIUS;

	// Variable pointing at top and bottom neighbouring location
	int e_ly_prev = e_ly - 1;
	int e_ly_next = e_ly + 1;

	// Variable pointing at left and right neighbouring location
	int e_lx_prev = e_lx - 1;
	int e_lx_next = e_lx + 1;

    float h = 1.0f / N;
    if (ix < (N + 2) && iy < (N + 2)) {
        if (ly < RADIUS) { // copy top and bottom halo
            //Copy Top Halo Element
		    if (blockIdx.y > 0) // Boundary Check
                d_v_s[lx + ly * (blockDim.x + 2 * RADIUS)] = d_v[tid - RADIUS * (N + 2)];

		    //Copy Bottom Halo Element
		    if (blockIdx.y < (gridDim.y - 1)) // Boundary Check
                d_v_s[lx + (e_ly + blockDim.y) * (blockDim.x + 2 * RADIUS)] = d_v[tid + blockDim.y * (N + 2)];
        }  

        if (lx < RADIUS) { // copy left and right halo
            // Copy Left Halo Element
            if (blockIdx.x > 0) // Boundary Check
                d_u_s[lx + ly * (blockDim.x + 2 * RADIUS)] = d_u[tid - RADIUS];
        
            // Copy Right Halo Element
            if (blockIdx.x < (gridDim.x - 1)) // Boundary Check
                d_u_s[(e_lx + blockDim.x) + ly * (blockDim.x + 2 * RADIUS)] = d_u[tid + blockDim.x];
        }

        // Copy Current Location
        d_v_s[lx + e_ly * (blockDim.x + 2 * RADIUS)] = d_v[tid];
        d_u_s[e_lx + ly * (blockDim.x + 2 * RADIUS)] = d_u[tid];
    }
    __syncthreads();

    if (ix >= 1 && ix <= N && iy >= 1 && iy <= N) {
        div_s[lTid] = -0.5f * h * (d_u_s[e_lx_next + ly * (blockDim.x + 2 * RADIUS)] - d_u_s[e_lx_prev + ly * (blockDim.x + 2 * RADIUS)] + 
            d_v_s[lx + e_ly_next * (blockDim.x + 2 * RADIUS)] - d_v_s[lx + e_ly_prev * (blockDim.x + 2 * RADIUS)]);
        p_s[lTid] = 0.0f;
        //printf("div[tid] %f (%d, %d)| d_u[tid + 1] %f | d_u[tid - 1] %f | d_v[tid + N + 2] %f | d_v[tid - N - 2] %f\n",
        //    div[tid], ix, iy, d_u[tid + 1], d_u[tid - 1], d_v[tid + N + 2], d_v[tid - N - 2]);
    }
    
    // Inefficient because multiple synchronization
    if (blockIdx.x == 0 || blockIdx.x == gridDim.x - 1 || blockIdx.y == 0 || blockIdx.y == gridDim.y - 1 &&
        ix < N + 2 && iy < N + 2) {
        __syncthreads(); // synchronization intra-block to ensure that the 
                         // threads in the boundary can compute correctly
        set_bndOnGPU(0, div_s, ix, iy, lTid);
        set_bndOnGPU(0, p_s, ix, iy, lTid);

        __syncthreads(); // now that all the borders are complete, we can compute
                         // corners
        set_crnOnGPU(0, div_s, ix, iy, lTid);
        set_crnOnGPU(0, p_s, ix, iy, lTid);
    }

    if (ix < (N + 2) && iy < (N + 2)) {
        div[tid] = div_s[lTid];
        p[tid] = p_s[lTid];
    }
}

// CUDA kernel to perform the LAST PROJECTION STEP (using Jacobi iteration outside of kernel)
__global__ void lastProjectOnGPU(float *d_u, float *d_v, float *p) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = ix + iy * (N + 2);
    int sizeShared = blockDim.x * blockDim.y;
    //int sizeSharedMax = (blockDim.x + 2) * (blockDim.y + 2);

    extern __shared__ float sharedMem[];
    float *d_u_s = (float*)sharedMem;
    float *d_v_s = (float*)&sharedMem[sizeShared];
    float *p_s = (float*)&sharedMem[2 * sizeShared];

    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int lTid = lx + ly * blockDim.x;

    // e_XX --> variables refers to expanded shared memory location in order to accomodate halo elements
	//Current Local ID with radius offset.
	int e_lx = lx + RADIUS;
	int e_ly = ly + RADIUS;

	// Variable pointing at top and bottom neighbouring location
	int e_ly_prev = e_ly - 1;
	int e_ly_next = e_ly + 1;

	// Variable pointing at left and right neighbouring location
	int e_lx_prev = e_lx - 1;
	int e_lx_next = e_lx + 1;

    float h = 1.0f / N;

    if (ix < (N + 2) && iy < (N + 2)) {
        if (ly < RADIUS) { // copy top and bottom halo
            //Copy Top Halo Element
		    if (blockIdx.y > 0) // Boundary Check
                p_s[e_lx + ly * (blockDim.x + 2 * RADIUS)] = p[tid - RADIUS * (N + 2)];

		    //Copy Bottom Halo Element
		    if (blockIdx.y < (gridDim.y - 1)) // Boundary Check
                p_s[e_lx + (e_ly + blockDim.y) * (blockDim.x + 2 * RADIUS)] = p[tid + blockDim.y * (N + 2)];
        }  

        if (lx < RADIUS) { // copy left and right halo
            // Copy Left Halo Element
            if (blockIdx.x > 0) // Boundary Check
                p_s[lx + e_ly * (blockDim.x + 2 * RADIUS)] = p[tid - RADIUS];
        
            // Copy Right Halo Element
            if (blockIdx.x < (gridDim.x - 1)) // Boundary Check
                p_s[(e_lx + blockDim.x) + e_ly * (blockDim.x + 2 * RADIUS)] = p[tid + blockDim.x];
        }

        // Copy Current Location
        p_s[e_lx + e_ly * (blockDim.x + 2 * RADIUS)] = p[tid];
        d_u_s[lTid] = d_u[tid];
        d_v_s[lTid] = d_v[tid];
    }
    __syncthreads();      


    if (ix >= 1 && ix <= N && iy >= 1 && iy <= N) {
        d_u_s[lTid] -= 0.5f * (p_s[e_lx_next + e_ly * (blockDim.x + 2 * RADIUS)] - p_s[e_lx_prev + e_ly * (blockDim.x + 2 * RADIUS)]) / h;
        d_v_s[lTid] -= 0.5f * (p_s[e_lx + e_ly_next * (blockDim.x + 2 * RADIUS)] - p_s[e_lx + e_ly_prev * (blockDim.x + 2 * RADIUS)]) / h;
    }
    
    // Inefficient because multiple synchronization
    if (blockIdx.x == 0 || blockIdx.x == gridDim.x - 1 || blockIdx.y == 0 || blockIdx.y == gridDim.y - 1 &&
        ix < N + 2 && iy < N + 2) {
        __syncthreads(); // synchronization intra-block to ensure that the 
                         // threads in the boundary can compute correctly
        set_bndOnGPU(1, d_u_s, ix, iy, lTid);
        set_bndOnGPU(2, d_v_s, ix, iy, lTid);

        __syncthreads(); // now that all the borders are complete, we can compute
                         // corners
        set_crnOnGPU(1, d_u_s, ix, iy, lTid);
        set_crnOnGPU(2, d_v_s, ix, iy, lTid);
    }

    if (ix < (N + 2) && iy < (N + 2)) {
        d_u[tid] = d_u_s[lTid];
        d_v[tid] = d_v_s[lTid];
    }
}

// Function to simulate the evolution of density
void dens_step(dim3 grid, dim3 block, float *d_x, float *d_x0, float *d_u, float *d_v, float *d_densTemp) {
    size_t sizeShared = block.x * block.y * sizeof(float);
    size_t sizeSharedMax = (block.x + 2 * hRADIUS) * (block.y + 2 * hRADIUS) * sizeof(float);
    
    add_sourceOnGPU<<<grid, block, 2 * sizeShared>>>(d_x, d_x0);

    float alpha = hDT * hDIFF * hN * hN;
    float beta = 1 + 4 * alpha;
    SWAP(d_x0, d_x);
    for (int k = 0; k < 40; k++) { // inefficient -> multiple kernel calls
        diffuseOnGPU<<<grid, block, 2 * sizeShared + sizeSharedMax>>>(0, d_x, d_x0, d_densTemp, alpha, beta);
        SWAP(d_densTemp, d_x);
    }
    
    SWAP(d_x0, d_x);
    advectOnGPU<<<grid, block, 7 * sizeShared>>>(0, d_x, d_x0, d_u, d_v);
}

// Function to simulate the evolution of velocity
void vel_step(dim3 grid, dim3 block, float *d_u, float *d_v, float *d_u0, float *d_v0, float *d_uTemp, float *d_vTemp) {
    size_t sizeShared = block.x * block.y * sizeof(float);
    size_t sizeSharedMax = (block.x + 2 * hRADIUS) * (block.y + 2 * hRADIUS) * sizeof(float);
    size_t sizeSharedU = (block.x + 2 * hRADIUS) * block.y * sizeof(float);
    size_t sizeSharedV = block.x * (block.y + 2 * hRADIUS) * sizeof(float);

    add_sourceOnGPU<<<grid, block, 2 * sizeShared>>>(d_u, d_u0);    
    add_sourceOnGPU<<<grid, block, 2 * sizeShared>>>(d_v, d_v0);

    SWAP(d_u, d_u0);
    SWAP(d_v, d_v0);
    /*
    cudaDeviceSynchronize();
    float *temp = (float*)malloc(sizeof(float) * (hN + 2) * (hN + 2));
    cudaMemcpy(temp, d_u0, sizeof(float) * (hN + 2) * (hN + 2), cudaMemcpyDeviceToHost);
    printDebug(temp);
    */

    float alpha = hDT * hVIS * hN * hN;
    float beta = 1 + 4 * alpha;
    for (int k = 0; k < 40; k++) { // inefficient -> multiple kernel calls
        diffuseOnGPU<<<grid, block, 2 * sizeShared + sizeSharedMax>>>(1, d_u, d_u0, d_uTemp, alpha, beta);
        diffuseOnGPU<<<grid, block, 2 * sizeShared + sizeSharedMax>>>(2, d_v, d_v0, d_vTemp, alpha, beta);
        SWAP(d_uTemp, d_u);
        SWAP(d_vTemp, d_v);
    }
    
    computeDivergenceAndPressureOnGPU<<<grid, block, 2 * sizeShared + sizeSharedU + sizeSharedV>>>(d_u, d_v, d_u0, d_v0);

    alpha = 1;
    beta = 4;
    // d_u0 is p, d_v0 is div
    for (int k = 0; k < 40; k++) { // inefficient -> multiple kernel calls
        diffuseOnGPU<<<grid, block, 2 * sizeShared + sizeSharedMax>>>(0, d_u0, d_v0, d_uTemp, alpha, beta);
        SWAP(d_uTemp, d_u0);
    }
    lastProjectOnGPU<<<grid, block, 2 * sizeShared + sizeSharedMax>>>(d_u, d_v, d_u0);

    SWAP(d_u0, d_u);
    SWAP(d_v0, d_v);
    advectOnGPU<<<grid, block, 7 * sizeShared>>>(1, d_u, d_u0, d_u0, d_v0);
    advectOnGPU<<<grid, block, 7 * sizeShared>>>(2, d_v, d_v0, d_u0, d_v0);

    computeDivergenceAndPressureOnGPU<<<grid, block,  2 * sizeShared + sizeSharedU + sizeSharedV>>>(d_u, d_v, d_u0, d_v0);
    // d_u0 is p, d_v0 is div
    for (int k = 0; k < 40; k++) { // inefficient -> multiple kernel calls
        diffuseOnGPU<<<grid, block, 2 * sizeShared + sizeSharedMax>>>(0, d_u0, d_v0, d_uTemp, alpha, beta);
        SWAP(d_uTemp, d_u0);
    }
    lastProjectOnGPU<<<grid, block, 2 * sizeShared + sizeSharedMax>>>(d_u, d_v, d_u0);
    
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

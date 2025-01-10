// 1:1

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>

// Constants
__constant__ int N;       // width and height of central grid
__constant__ float DT;    // timestep
__constant__ float VIS;   // viscosity
__constant__ float DIFF;  // diffusion rate
__constant__ float a1;    // parameter for the diffusion Jacobi
__constant__ float a2;    // parameter for the projection Jacobi

// Function to get correct grid index
#define IX(x, y) ((x) + (y) * (N + 2))

// Function to SWAP two pointers
#define SWAP(x0, x) \
{ \
    float *tmp = x0; \
    x0 = x; \
    x = tmp; \
}

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

// Function to measure time in seconds
double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

// CUDA kernel for setting boundary conditions - naive version
__device__ void set_bndOnGPU(int b, float *d_x) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int tid = ix + iy * (N + 2);

    if (ix == 0 && iy < N + 1 && iy > 0)
        d_x[tid] = b == 1 ? -d_x[tid + 1] : d_x[tid + 1];
    else if (ix == N + 1 && iy < N + 1 && iy > 0)
        d_x[tid] = b == 1 ? -d_x[tid - 1] : d_x[tid - 1];
    else if (iy == 0 && ix > 0 && ix < N + 1)
        d_x[tid] = b == 2 ? -d_x[tid + N + 2] : d_x[tid + N + 2];
    else if (iy == N + 1 && ix > 0 && ix < N + 1)
        d_x[tid] = b == 2 ? -d_x[tid - N - 2] : d_x[tid - N - 2];

    if (ix == 0 && iy == 0)
        d_x[tid] = 0.5f * (d_x[tid + 1] + d_x[tid + N + 2]);
    else if (ix == 0 && iy == N + 1)
        d_x[tid] = 0.5f * (d_x[tid + 1] + d_x[tid - N - 2]);
    else if (ix == N + 1 && iy == 0)
        d_x[tid] = 0.5f * (d_x[tid - 1] + d_x[tid + N + 2]);
    else if (ix == N + 1 && iy == N + 1) 
        d_x[tid] = 0.5f * (d_x[tid - 1] + d_x[tid - N - 2]);
}

// CUDA kernel for adding sources - naive version
__global__ void add_sourceOnGPU(float *d_x, float *d_s) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * (N + 2);

    if (idx < 0 || (idx > ((N + 2) * (N + 2))))
        printf("ADDING SOURCE - idx: %d, ix: %d, iy: %d\n", idx, ix, iy);

    if(ix < (N + 2) && iy < (N + 2))
        d_x[idx] += DT * d_s[idx];
}

// CUDA kernel for diffusion step - naive version
__global__ void diffuseOnGPU(int b, float *d_x, float *d_x0, float *d_x_new, int whichAlpha, int k) {
    // In questa implementazione i thread ai confini sono idle

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * (N + 2);

    if (idx < 0 || (idx > ((N + 2) * (N + 2))))
        printf("DIFFUSION - idx: %d, ix: %d, iy: %d, k: %d\n", idx, ix, iy, k);


    // N da passare
    //float a = dt * diff * N * N; // si puo passare prima?
    float a = !whichAlpha ? a1 : a2;

    if (ix >= 1 && iy >= 1 && ix <= N && iy <= N) {
        int left = idx - 1;
        int right = idx + 1;
        int top = idx + N + 2;
        int bot = idx - N - 2;
        // serve una variabile tipo x_new ?
        d_x_new[idx] = (d_x0[idx] + a * (d_x[left] + d_x[right] + d_x[bot] + d_x[top])) / (1 + 4 * a); // RIGUARDARE !!!!!
    } else if (ix < (N + 2) && iy < (N + 2)) {
        __syncthreads();
        set_bndOnGPU(b, d_x);
    }
 

    // set_bnd ?
}

// CUDA kernel for advection step - naive version
__global__ void advectOnGPU (int b, float *d_d, float *d_d0, float *d_u, float *d_v) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * (N + 2);

    if (idx < 0 || (idx > ((N + 2) * (N + 2))))
        printf("ADVECTION - idx: %d, ix: %d, iy: %d\n", idx, ix, iy);


    int i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    dt0 = DT * N;
    if (ix >= 1 && iy >= 1 && ix <= N && iy <= N) {
        x = ix - dt0 * d_u[idx];
        y = iy - dt0 * d_v[idx];

        if (x < 0.5)
            x = 0.5;
        if (x > N + 0.5)
            x = N + 0.5;
        i0 = (int)x;
        i1 = i0 + 1;

        if (y < 0.5)
            y = 0.5;
        if (y > N + 0.5)
            y = N + 0.5;
        j0 = (int)y;
        j1 = j0 + 1;

        s1 = x - i0;
        s0 = 1 - s1;
        t1 = y - j0;
        t0 = 1 - t1;

        d_d[idx] = s0 * (t0 * d_d0[IX(i0, j0)] + t1 * d_d0[IX(i0, j1)]) +
            s1 * (t0 * d_d0[IX(i1, j0)] + t1 * d_d0[IX(i1, j1)]);
    } else if (ix < (N + 2) && iy < (N + 2)) {
        __syncthreads();
        set_bndOnGPU(b, d_d);
    }

    // set_bnd ?
}

// CUDA kernel for initializing pressure and divergence
__global__ void computeDivergenceAndPressureOnGPU(int b, float *d_u, float *d_v, float *p, float *div) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * (N + 2);

    if (idx < 0 || (idx > ((N + 2) * (N + 2))))
        printf("DIVERGENCE AND PRESSURE - idx: %d, ix: %d, iy: %d\n", idx, ix, iy);


    float h = 1.0f / N;
    
    if (ix >= 1 && iy >= 1 && ix <= N && iy <= N) {
        div[idx] = -0.5f * h * (d_u[idx + 1] - d_u[idx - 1] + d_v[idx + N + 2] - d_v[idx - N - 2]);
        p[idx] = 0.0f;
    } else if (ix < (N + 2) && iy < (N + 2)) {
        __syncthreads();
        set_bndOnGPU(b, p);
        set_bndOnGPU(b, div);
    }
}

// CUDA kernel for projection step - naive version
__global__ void projectOnGPU(float *d_u, float *d_v, float *p) {
    // chiamare diffuse con jacobi (in host come kernel call)

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * (N + 2);

    if (idx < 0 || (idx > ((N + 2) * (N + 2))))
        printf("PROJECTION - idx: %d, ix: %d, iy: %d\n", idx, ix, iy);

    float h = 1.0f / N;

    // manca if ??
    if (ix >= 1 && iy >= 1 && ix <= N && iy <= N) {
        d_u[idx] -= 0.5f * (p[idx + 1] - p[idx - 1]) / h;
        d_v[idx] -= 0.5f * (p[idx + N + 2] - p[idx - N - 2]) / h;
    } else if (ix < (N + 2) && iy < (N + 2)) {
        __syncthreads();
        set_bndOnGPU(1, d_u);
        set_bndOnGPU(2, d_v);
    }

    // set_bnd ?
}

// dens_step - naive version
void dens_step(dim3 grid, dim3 block, float *d_x, float *d_x0, float *d_u, float *d_v, float *d_dens_new) {
    // MANCANO TUTTI I BOUND -> fatti ?

    add_sourceOnGPU<<<grid, block>>>(d_x, d_x0);
    cudaDeviceSynchronize();

    SWAP(d_x0, d_x);
    for (int k = 0; k < 40; k++) {
        diffuseOnGPU<<<grid, block>>>(0, d_x, d_x0, d_dens_new, 0, k);
        cudaDeviceSynchronize();
        //set_bndOnGPU<<<grid, block>>>(0, d_x);
        //cudaDeviceSynchronize();
        SWAP(d_x, d_dens_new);
    }  

    SWAP(d_x0, d_x);
    advectOnGPU<<<grid, block>>>(0, d_x, d_x0, d_u, d_v);
    cudaDeviceSynchronize();
    //set_bndOnGPU<<<grid, block>>>(0, d_x);
    //cudaDeviceSynchronize();
}

// vel_step - naive version
void vel_step(dim3 grid, dim3 block, float *d_u, float *d_v, float *d_u0, float *d_v0, float *d_u_new, float *d_v_new) {
    // MANCANO TUTTI I BOUND -> fatti ?
    // CREARE KERNEL PER SWAPPARE I PUNTATORI ? -> fatto ?

    add_sourceOnGPU<<<grid, block>>>(d_u, d_u0);
    add_sourceOnGPU<<<grid, block>>>(d_v, d_v0);
    cudaDeviceSynchronize();

    SWAP(d_u0, d_u); // funziona ? -> probabilmente no
    SWAP(d_v0, d_v); // funziona ?
    for (int k = 0; k < 40; k++) {
        diffuseOnGPU<<<grid, block>>>(1, d_u, d_u0, d_u_new, 0, k);
        diffuseOnGPU<<<grid, block>>>(2, d_v, d_v0, d_v_new, 0, k); 
        cudaDeviceSynchronize();
        //set_bndOnGPU<<<grid, block>>>(1, d_u);
        //set_bndOnGPU<<<grid, block>>>(2, d_v);
        //cudaDeviceSynchronize();
        SWAP(d_u, d_u_new);
        SWAP(d_v, d_v_new);
    }
    computeDivergenceAndPressureOnGPU<<<grid, block>>>(0, d_u, d_v, d_u0, d_v0); // divergence -> d_v0, pressure -> d_u0
    cudaDeviceSynchronize();
    //set_bndOnGPU<<<grid, block>>>(0, d_u0);
    //set_bndOnGPU<<<grid, block>>>(0, d_v0);
    //cudaDeviceSynchronize();
    for (int k = 0; k < 40; k++) {
        diffuseOnGPU<<<grid, block>>>(0, d_u0, d_v0, d_u_new, 1, k);
        cudaDeviceSynchronize();
        //set_bndOnGPU<<<grid, block>>>(0, d_u0);
        //cudaDeviceSynchronize();
        SWAP(d_u0, d_u_new)
    }
    projectOnGPU<<<grid, block>>>(d_u, d_v, d_u0);
    cudaDeviceSynchronize();
    //set_bndOnGPU<<<grid, block>>>(1, d_u);
    //set_bndOnGPU<<<grid, block>>>(2, d_v);
    //cudaDeviceSynchronize();

    SWAP(d_u0, d_u); // funziona ?
    SWAP(d_v0, d_v); // funziona ?
    advectOnGPU<<<grid, block>>>(1, d_u, d_u0, d_u0, d_v0);
    advectOnGPU<<<grid, block>>>(2, d_v, d_v0, d_u0, d_v0);
    cudaDeviceSynchronize();
    //set_bndOnGPU<<<grid, block>>>(1, d_u);
    //set_bndOnGPU<<<grid, block>>>(2, d_v);
    //cudaDeviceSynchronize();

    computeDivergenceAndPressureOnGPU<<<grid, block>>>(0, d_u, d_v, d_u0, d_v0); // divergence -> d_v0, pressure -> d_u0
    cudaDeviceSynchronize();
    //set_bndOnGPU<<<grid, block>>>(0, d_u0);
    //set_bndOnGPU<<<grid, block>>>(0, d_v0);
    //cudaDeviceSynchronize();
    for (int k = 0; k < 40; k++) {
        diffuseOnGPU<<<grid, block>>>(0, d_u0, d_v0, d_u_new, 1, k);
        cudaDeviceSynchronize();
        //set_bndOnGPU<<<grid, block>>>(0, d_u0);
        //cudaDeviceSynchronize();
        SWAP(d_u0, d_u_new)
    }
    projectOnGPU<<<grid, block>>>(d_u, d_v, d_u0);
    cudaDeviceSynchronize();
    //set_bndOnGPU<<<grid, block>>>(1, d_u);
    //set_bndOnGPU<<<grid, block>>>(2, d_v);
    //cudaDeviceSynchronize();
}

// Copied from the sequential implementation
void initializeParameters(float *dens, float *dens_prev, float *u, float *u_prev, float *v, float *v_prev, int hN) {
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

void printSomething(float *dens, float *u, float *v, int hN) {
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

void checkStability(float *u, float *v, int hN) {
    int i, j, h = 1.0 / hN;
    float error = 1.0E-8;
    float found = 0;
    for (i = 0; i < hN + 2; i++) {
        for (j = 0; j < hN + 2; j++) {

            // PROBLEMA INDICE QUI

            float divergence = 0.5 * h * ((u[j + 1 + i * (hN + 2)] - u[j - 1 + i * (hN + 2)]) + 
                v[j + (i + 1) * (hN + 2)] - v[j + (i - 1) * (hN + 2)]);
            if (divergence > error) {
                printf("The velocity is not divergence-free\n");
                found = 1;
            }
        }
    }

    found ? printf("all bad\n") : printf("all good\n");

}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    // Set CUDA device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Check command line arguments
    if (argc != 3) {
        printf("Usage: %s <block_dim_x> <block_dim_y>\n", argv[0]);
        return 1;
    }

    // Parse command line arguments
    int block_dim_x = atoi(argv[1]);
    int block_dim_y = atoi(argv[2]);

    int hN = (1<<3) - 2;
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

    float hDT = 0.016f, hVIS = 0.0025f, hDIFF = 0.1f;
    float ha1 = hDT * hDIFF * hN * hN, ha2 = 1.0f;


    // Initialize Parameters
    /*
    initializeParameters(dens, dens_prev, u, u_prev, v, v_prev);
    printf("HELLO init\n");
    printSomething(dens_prev, u_prev, v_prev);
    printf("Hello end\n");  
    */

    // Allocate device memory
    float *d_u, *d_u_prev, *d_v, *d_v_prev, *d_dens, *d_dens_prev, *d_u_new, *d_v_new, *d_dens_new;
    CHECK(cudaMalloc((void **)&d_u, nBytes));
    CHECK(cudaMalloc((void **)&d_u_prev, nBytes));
    CHECK(cudaMalloc((void **)&d_v, nBytes));
    CHECK(cudaMalloc((void **)&d_v_prev, nBytes));
    CHECK(cudaMalloc((void **)&d_dens, nBytes));
    CHECK(cudaMalloc((void **)&d_dens_prev, nBytes));
    CHECK(cudaMalloc((void **)&d_u_new, nBytes));
    CHECK(cudaMalloc((void **)&d_v_new, nBytes));
    CHECK(cudaMalloc((void **)&d_dens_new, nBytes));


    CHECK(cudaMemcpyToSymbol(N, &hN, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(DT, &hDT, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(VIS, &hVIS, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(DIFF, &hDIFF, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(a1, &ha1, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(a2, &ha2, sizeof(float)));


    // Transfer data from host to device
    /*
    CHECK(cudaMemcpy(d_u, u, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_u_prev, u_prev, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_v, v, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_v_prev, v_prev, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dens, dens, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dens_prev, dens_prev, nBytes, cudaMemcpyHostToDevice));
    */

    // Define grid and block dimensions
    dim3 block, grid;
    block = dim3(block_dim_x, block_dim_y);
    grid = dim3(((hN + 2) + block.x - 1) / block.x, ((hN + 2) + block.y - 1) / block.y);

    // Launch kernel
    int z = 0;
    int first = 1;
    iStart = cpuSecond();
    while (z++ < 5) {
        if (first) {
            initializeParameters(dens, dens_prev, u, u_prev, v, v_prev, hN);
            first = 0;
            printf("HELLO init\n");
            printSomething(dens_prev, u_prev, v_prev, hN);
            printf("Hello end\n");

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


        vel_step(grid, block, d_u, d_v, d_u_prev, d_v_prev, d_u_new, d_v_new);
        dens_step(grid, block, d_dens, d_dens_prev, d_u, d_v, d_dens_new);

        // COPIARE DA DEVICE A HOST !!!!
        CHECK(cudaDeviceSynchronize());

        // DA METTERE NEL CICLO SOLO PER DEBUG
        
        //if (z == 50) {
        printf("DEBUG | z = %d\n", z);
        CHECK(cudaMemcpy(u, d_u, nBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(u_prev, d_u_prev, nBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(v, d_v, nBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(v_prev, d_v_prev, nBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(dens, d_dens, nBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(dens_prev, d_dens_prev, nBytes, cudaMemcpyDeviceToHost));


        printSomething(dens, u, v, hN);
        //checkStability(u, v, hN);
        //}
        
    }

    CHECK(cudaMemcpy(u, d_u, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(v, d_v, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(dens, d_dens, nBytes, cudaMemcpyDeviceToHost));

    //printSomething(dens, u, v, hN);
    //checkStability(u, v, hN);

    iElaps = cpuSecond() - iStart;
    printf("grid: %d, <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", size, grid.x, grid.y, block.x, block.y, iElaps);

    printSomething(dens, u, v, hN);


    // Cleaning
    free(u); free(u_prev);
    free(v); free(v_prev);
    free(dens); free(dens_prev);

    CHECK(cudaFree(d_u)); CHECK(cudaFree(d_u_prev));
    CHECK(cudaFree(d_v)); CHECK(cudaFree(d_v_prev));
    CHECK(cudaFree(d_dens)); CHECK(cudaFree(d_dens_prev));
    CHECK(cudaFree(d_u_new)); CHECK(cudaFree(d_v_new)); CHECK(cudaFree(d_dens_new));

    return 0;
}
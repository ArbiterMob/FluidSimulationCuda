#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h>
#include <stdint.h>

#define SSE_DATA_LANE 16
#define ELEM 4

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

const int N = 14;            // (WidthGrid - 2) 
const float DT = 0.016f;    // Instantaneous change in time (timestep)
const float VIS = 0.0025f;    // Viscosity coefficient
const float DIFF = 0.1f;      // Diffusion coefficient
int totIterations, iterationsX;

// SWAP macro
#define SWAP(x0, x) {float *tmp = x0; x0 = x; x = tmp;}
#define SWAP_SIMD(x0, x) {__m128 *tmp = x0; x0 = x; x = tmp;}

// Function to debug a float grid
void printDebug(float *x) {
    int i, j;

    for (i = 0; i < N + 2; i++) {
        for (j = 0; j < N + 2; j++) {
            printf("[%f] ", x[j + i * (N + 2)]);
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
    for (i = 0; i < N + 2; i++) {
        for (j = 0; j < N + 2; j++) {
            printf("[%f] ", dens[j + i * (N +2)]);
        }
        printf("\n");
    }
    printf("\n\n");

    printf("VELOCITY\n");
    for (i = 0; i < N + 2; i++) {
        for (j = 0; j < N + 2; j++) {
            printf("[%f, %f] ", u[j + i * (N +2)], v[j + i * (N +2)]);
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

// Function to set the boundary conditions
void set_bnd(int b, float *x) {
    int i;

    for (i = 1; i <= N; i++) {
        x[i * (N + 2)] = b == 1 ? -x[1 + i * (N + 2)] : x[1 + i * (N + 2)];
        x[(N + 1) + i * (N + 2)] = b == 1 ? -x[N + i * (N + 2)] : x[N + i * (N + 2)];
        x[i] = b == 2 ? -x[i + 1 * (N + 2)] : x[i + 1 * (N + 2)];
        x[i + (N + 1) * (N + 2)] = b == 2 ? -x[i + N * (N + 2)] : x[i + N * (N + 2)];
    }
    x[0] = 0.5f * (x[1] + x[1 * (N + 2)]);
    x[(N + 1) * (N + 2)] = 0.5f * (x[1 + (N + 1) * (N + 2)] + x[N * (N + 2)]);
    x[N + 1] = 0.5f * (x[N] + x[(N + 1) + 1 * (N + 2)]);
    x[(N + 1) + (N + 1) * (N + 2)] = 0.5f * (x[N + (N + 1) * (N + 2)] + x[(N + 1) + N * (N + 2)]);
}

// Function to ADD EXTERNAL SOURCES
void add_source(float *x, float *s) {
    int i, size = (N + 2) * (N + 2);
    __m128 xmm1, xmm2, xmm3;
    __m128 *x_simd = (__m128*)x;
    __m128 dt_simd = _mm_set1_ps(DT);
    float test[4];

    for (i = 0; i < totIterations; i++) {
        //xmm1 = _mm_load_ps((float*)(x_simd + i));
        //xmm2 = _mm_load_ps((float*)(s_simd + i));
        xmm1 = _mm_load_ps(x + i * ELEM);
        xmm2 = _mm_load_ps(s + i * ELEM);

        /*
        printf("xmm1 %d\n", i);
        _mm_store_ps(test, xmm1);
        for(int j = 0; j < 4; j++) {
            printf("[%f] ", test[j]);
        }
        printf("\n");
        printf("---------\n");
        printf("xmm2 %d\n", i);
        _mm_store_ps(test, xmm2);
        for(int j = 0; j < 4; j++) {
            printf("[%f] ", test[j]);
        }
        printf("\n");
        printf("------------------------------\n");
        */
        
        xmm3 = _mm_mul_ps(xmm2, dt_simd);
        x_simd[i] = _mm_add_ps(xmm1, xmm3);

        /*
        printf("x_simd[i] %d\n", i);
        _mm_store_ps(test, x_simd[i]);
        for(int j = 0; j < 4; j++) {
            printf("[%f] ", test[j]);
        }
        printf("\n");
        */
        
    }
}

// Function to perform DIFFUSION (using Jacobi iteration)
void diffuse (int b, float *x, float *x0, float alpha, float beta) {
    int i, j, k, size = (N + 2) * (N + 2);
    //float a = DT * diff * N * N; // coefficient for the jacobi solution
    float *x_new = (float*)malloc(sizeof(float) * size); // temp grid to update values 
                                                         // after every jacobi iteration
    
    __m128 *x_new_simd = (__m128*)x_new;
    __m128 *x_simd = (__m128*)x;
    __m128 xmm_x0, xmm_x, xmm_right, xmm_left, xmm_top, xmm_bot, xmm_temp;
    __m128i mask_left = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0);
    __m128i mask_right = _mm_set_epi32(0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    __m128 xmm_alphaFracBeta = _mm_set1_ps(alpha / beta);
    __m128 xmm_fracBeta = _mm_set1_ps(1 / beta);
    float test[4];
    float temp_right;
    
    for (k = 0; k < 40; k++) {
        printf("x %d\n", k);
        printDebug(x);
        for (i = 0; i < N + 2; i++) {
            for (j = 0; j < iterationsX; j++) {
                if (i > 0 && i < N + 1) {
                    xmm_x = _mm_load_ps(x + (j + i * iterationsX) * ELEM);

                    xmm_x0 = _mm_load_ps(x0 + (j + i * iterationsX) * ELEM);
                    xmm_x0 = _mm_mul_ps(xmm_x0, xmm_fracBeta);

                    xmm_top = _mm_load_ps(x + (j + (i - 1) * iterationsX) * ELEM);
                    //xmm_top = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(xmm_top), (__m128i)mask));
                    xmm_top = _mm_mul_ps(xmm_top, xmm_alphaFracBeta);

                    xmm_bot = _mm_load_ps(x + (j + (i + 1) * iterationsX) * ELEM);
                    //xmm_bot = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(xmm_bot), (__m128i)mask));
                    xmm_bot = _mm_mul_ps(xmm_bot, xmm_alphaFracBeta);

                    // DISTINGURE TRA CENTRALI E BORDI !!!                    

                    if (j < iterationsX - 1) {
                        xmm_temp = _mm_load_ps(x + ((j + 1) + i * iterationsX) * ELEM);
                        
                        xmm_left = _mm_shuffle_ps(xmm_x, xmm_x, 0x90);
                        xmm_left = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(xmm_left), (__m128i)mask_left));
                        
                        printf("xmm_left %d\n", j + i * iterationsX);
                        _mm_store_ps(test, xmm_left);
                        for(int j = 0; j < 4; j++) {
                            printf("[%f] ", test[j]);
                        }
                        printf("\n");
                        printf("---------\n");
                        
                        xmm_left = _mm_mul_ps(xmm_left, xmm_alphaFracBeta);

                        xmm_right = _mm_insert_ps(_mm_shuffle_ps(xmm_x, xmm_x, 0x38), xmm_temp, 0x30);
                        xmm_right = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(xmm_right), (__m128i)mask_left));
                        
                        printf("xmm_right %d\n", j + i * iterationsX);
                        _mm_store_ps(test, xmm_right);
                        for(int j = 0; j < 4; j++) {
                            printf("[%f] ", test[j]);
                        }
                        printf("\n");
                        printf("-------------------------------\n");
                        
                        xmm_right = _mm_mul_ps(xmm_right, xmm_alphaFracBeta);

                    }

                    if (j > 0) {
                        xmm_temp = _mm_load_ps(x + ((j - 1) + i * iterationsX) * ELEM);

                        xmm_left = _mm_insert_ps(_mm_shuffle_ps(xmm_x, xmm_x, 0x90), xmm_temp, 0xC0);
                        xmm_left = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(xmm_left), (__m128i)mask_right));
                        
                        printf("xmm_left %d\n", j + i * iterationsX);
                        _mm_store_ps(test, xmm_left);
                        for(int j = 0; j < 4; j++) {
                            printf("[%f] ", test[j]);
                        }
                        printf("\n");
                        printf("---------\n");
                        
                        xmm_left = _mm_mul_ps(xmm_left, xmm_alphaFracBeta);

                        xmm_right = _mm_shuffle_ps(xmm_x, xmm_x, 0x39);
                        xmm_right = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(xmm_right), (__m128i)mask_right));
                        
                        printf("xmm_right %d\n", j + i * iterationsX);
                        _mm_store_ps(test, xmm_right);
                        for(int j = 0; j < 4; j++) {
                            printf("[%f] ", test[j]);
                        }
                        printf("\n");
                        printf("-------------------------------\n");
                        
                        xmm_right = _mm_mul_ps(xmm_right, xmm_alphaFracBeta);

                    }

                    x_new_simd[j + i * iterationsX] = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(xmm_left, xmm_right), xmm_top), xmm_bot), xmm_x0);
                }
            }
        }
        SWAP(x, x_new);
        SWAP_SIMD(x_simd, x_new_simd);
        set_bnd(b, x);
    }
    free(x_new);
    
    /*
    for (k = 0; k < 40; k++) {
        for (i = 1; i <= N; i++) {
            for (j = 1; j <= N; j++) {
                x_new[j + i * (N + 2)] = (x0[j + i * (N + 2)] + alpha * (x[(j - 1) + i * (N + 2)] + x[(j + 1) + i * (N + 2)] + x[j + (i - 1) * (N + 2)] +
                    x[j + (i + 1) * (N + 2)])) / beta;
            }
        }
        SWAP(x, x_new); // update the grid
        set_bnd(b, x);
    }
    free(x_new);
    */
}

// Function to perform ADVECTION (using bilinear interpolation)
void advect (int b, float *d, float *d0, float *u, float *v) {
    int i, j, i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    dt0 = DT * N; // N is the scale factor
    /* DA RIGUARDARE MA NON CREDO SIA FATTIBILE CON SIMD (o meglio, è possibile, ma il codice diventa lungo e forse meno efficiente)
    __m128 *d_simd = (__m128*)d;
    __m128 *u_simd = (__m128*)u;
    __m128 *v_simd = (__m128*)v;
    __m128 dt0_simd = _mm_set1_ps(dt0);
    __m128 xmm_clamp_min;
    __m128 xmm_clamp_max;
    __m128 xmm_x, xmm_y, xmm_index_x, xmm_index_y;

    for (i = 0; i < N + 2; i++) {
        for (j = 0; j < iterationsX; j++) {
            if (i > 0 && i < N + 1) {
                xmm_index_x = _mm_set_ps(j, j + 1, j + 2, j + 3);
                xmm_index_y = _mm_set1_ps(i);

                xmm_x = _mm_sub_ps(xmm_index_x, _mm_mul_ps(dt0_simd, u_simd[j + i * iterationsX]));
                xmm_y = _mm_sub_ps(xmm_index_y, _mm_mul_ps(dt0_simd, v_simd[j + i * iterationsX]));

                // if x < 0.5 -> clamp_min
                // if x > N + 0.5 -> clamp_max

                // if y < 0.5
                // if y > N + 0.5   
            }
        }
    }
    */

    for (i = 1; i <= N; i++) {
        for (j = 1; j <= N; j++) {
            x = j - dt0 * u[j + i * (N + 2)]; 
            y = i - dt0 * v[j + i * (N + 2)];

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

            d[j + i * (N + 2)] = s0 * (t0 * d0[j0 + i0 * (N + 2)] + t1 * d0[j0 + i1 * (N + 2)]) +
                s1 * (t0 * d0[j1 + i0 * (N + 2)] + t1 * d0[j1 + i1 * (N + 2)]);
        } 
    }
    set_bnd(b, d);
}

// Function to perform PROJECTION (using jacobi iteration)
void project(float *u, float *v, float *p, float *div) {
    int i, j, k, size = (N + 2) * (N + 2);
    float h, alpha, beta;
    float *p_new = (float*)malloc(sizeof(float) * size);

    h = 1.0f / N;
    for (i = 1; i <= N; i++) {
        for (j = 1; j <= N; j++) {
            div[j + i * (N + 2)] = -0.5f * h * (u[(j + 1) + i * (N + 2)] - u[(j - 1) + i * (N + 2)] +
                v[j + (i + 1) * (N + 2)] - v[j + (i - 1) * (N + 2)]);
            p[j + i * (N + 2)] = 0.0f;
        }
    }
    set_bnd(0, div);
    set_bnd(0, p);

    alpha = 1;
    beta = 4;
    diffuse(0, p, div, alpha, beta);

    for (i = 1; i <= N; i++) {
        for (j = 1; j <= N; j++) {
            u[j + i * (N + 2)] -= 0.5f * (p[(j + 1) + i * (N + 2)] - p[(j - 1) + i * (N + 2)]) / h;
            v[j + i * (N + 2)] -= 0.5f * (p[j + (i + 1) * (N + 2)] - p[j + (i - 1) * (N + 2)]) / h;
        }
    }
    set_bnd(1, u);
    set_bnd(2, v);
}

// Function to simulate the evolution of density
void dens_step(float *x, float *x0, float *u, float *v, float diff) {
    add_source(x, x0);

    float alpha = DT * diff * N * N;
    float beta = 1 + 4 * alpha;
    SWAP(x0, x); 
    diffuse(0, x, x0, alpha, beta);

    SWAP(x0, x); 
    advect(0, x, x0, u, v);
}

// Function to simulate the evolution of velocity
void vel_step(float *u, float *v, float *u0, float *v0, float visc) {
    add_source(u, u0);
    add_source(v, v0);

    float alpha = DT * visc * N * N;
    float beta = 1 + 4 * alpha;
    SWAP(u0, u);
    diffuse(1, u, u0, alpha, beta);
    SWAP(v0, v);
    diffuse(2, v, v0, alpha, beta);
    project(u, v, u0, v0);

    SWAP(u0, u);
    SWAP(v0, v);
    advect(1, u, u0, u0, v0);
    advect(2, v, v0, u0, v0);
    project(u, v, u0, v0);
}

// Function to initialize the density and velocity
void initializeParameters(float *dens, float *dens_prev, float *u, float *u_prev, float *v, float *v_prev) {
    int i, j;
    int center_x = (N + 2) / 2, center_y = (N + 2) / 2;
    int radius = (N + 2) / 8;

    // density source
    for (i = 0; i < N + 2; i++) {
        for (j = 0; j < N + 2; j++) {
            if ((j < center_x + radius) && (j >= center_x - radius) &&
                (i < center_y + radius) && (i >= center_y - radius))
                dens_prev[j + i * (N + 2)] = (rand() % 100) / 1000.0f;
            else 
                dens_prev[j + i * (N + 2)] = 0.0f;
            dens[j + i * (N + 2)] = 0.0f;
        }
    } 

    // velocity
    for (i = 0; i < N + 2; i++) {
        for (j = 0; j < N + 2; j++) {
            u_prev[j + i * (N +2)] = (rand() % 100) / 100.0f;
            v_prev[j + i * (N +2)] = (rand() % 100) / 100.0f;

            u[j + i * (N +2)] = 0.0f;
            v[j + i * (N +2)] = 0.0f;
        }
    }
}

int main(int argc, char **argv) {
    int size = (N + 2) * (N + 2);
    int nBytes = ((size * sizeof(float) + SSE_DATA_LANE - 1) / SSE_DATA_LANE) * SSE_DATA_LANE;
    totIterations = (size + ELEM - 1) / ELEM;
    iterationsX = (N + 2 + ELEM - 1) / ELEM;
    printf("size %d, nBytes %d, totIterations %d\n", size, nBytes, totIterations);

    float *u = (float*)_mm_malloc(nBytes, SSE_DATA_LANE);
    float *u_prev = (float*)_mm_malloc(nBytes, SSE_DATA_LANE);
    float *v = (float*)_mm_malloc(nBytes, SSE_DATA_LANE);
    float *v_prev = (float*)_mm_malloc(nBytes, SSE_DATA_LANE);
    float *dens = (float*)_mm_malloc(nBytes, SSE_DATA_LANE);
    float *dens_prev = (float*)_mm_malloc(nBytes, SSE_DATA_LANE);

    uint64_t clock_counter_start;
    uint64_t clock_counter_end;

    double iStart, iElaps;

    /*
    __m128 *u_simd = (__m128*)u;
    __m128 *u_prev_simd = (__m128*)u_prev;
    __m128 *v_simd = (__m128*)v;
    __m128 *v_prev_simd = (__m128*)v_prev;
    __m128 *dens_simd = (__m128*)dens;
    __m128 *dens_prev_simd = (__m128*)dens_prev;
    */


    int z = 0;
    int first = 1;
    iStart = cpuSecond();
    clock_counter_start = __rdtsc();
    while (z++ < 1) {
        if (first) {
            initializeParameters(dens, dens_prev, u, u_prev, v, v_prev);
            first = 0;
            //printf("HELLO init\n");
            //printStateGrid(dens_prev, u_prev, v_prev);
            //printf("Hello end\n");
        } else {
            for (int i = 0; i < size; i++) {
                u_prev[i] = 0.0f;
                v_prev[i] = 0.0f;
                dens_prev[i] = 0.0f;
            }
        }

        vel_step(u, v, u_prev, v_prev, VIS);
        dens_step(dens, dens_prev, u, v, DIFF);

        //printStateGrid(dens, u, v);
        //checkStability(u, v);
    }
    clock_counter_end = __rdtsc();
    iElaps = cpuSecond() - iStart;
    printf("nElements %d - elapsed %f sec | %ld cycles\n", size, iElaps, clock_counter_end - clock_counter_start);

    printStateGrid(dens, u, v);

    // Cleaning
    free(u); free(u_prev);
    free(v); free(v_prev);
    free(dens); free(dens_prev);

    return 0;
}
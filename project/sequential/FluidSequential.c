#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 8190      // (WidthGrid - 2) 
#define DT 0.016f   // Instantaneous change in time (timestep)
#define VIS 0.0025f // Viscosity coefficient
#define DIFF 0.1f   // Diffusion coefficient
#define Z 50        // Number of Steps of Simulation (used for mean runtime)

// SWAP macro
#define SWAP(x0, x) {float *tmp = x0; x0 = x; x = tmp;}

// Global variables
double arrayElaps[Z], timeAdvection[Z], timeDiffusion[Z], timeDivergence[Z], timeProjection[Z], timeSource[Z];

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
    for (i = 0; i < size; i++)
        x[i] += DT * s[i];
}

// Function to perform DIFFUSION (using Jacobi iteration)
void diffuse (int b, float *x, float *x0, float alpha, float beta) {
    int i, j, k, size = (N + 2) * (N + 2);
    //float a = DT * diff * N * N; // coefficient for the jacobi solution
    float *x_new = (float*)malloc(sizeof(float) * size); // temp grid to update values 
                                                         // after every jacobi iteration

    for (k = 0; k < 40; k++) {
        //double temp = cpuSecond();
        for (i = 1; i <= N; i++) {
            for (j = 1; j <= N; j++) {
                x_new[j + i * (N + 2)] = (x0[j + i * (N + 2)] + alpha * (x[(j - 1) + i * (N + 2)] + x[(j + 1) + i * (N + 2)] + x[j + (i - 1) * (N + 2)] +
                    x[j + (i + 1) * (N + 2)])) / beta;
            }
        }
        //printf("diffuse %d - %f\n", k, cpuSecond() - temp);
        SWAP(x, x_new); // update the grid
        set_bnd(b, x);
    }
    free(x_new);
}

// Function to perform ADVECTION (using bilinear interpolation)
void advect (int b, float *d, float *d0, float *u, float *v) {
    int i, j, i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    dt0 = DT * N; // N is the scale factor
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

void computeDivergenceAndPressure(float *u, float *v, float *p, float *div) {
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
}

// Function to perform PROJECTION (using jacobi iteration)
void lastProject(float *u, float *v, float *p, float *div) {
    int i, j;

    float h = 1.0f / N;
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
void vel_step(float *u, float *v, float *u0, float *v0, float visc, int z) {
    double iStart, iElaps;

    iStart = cpuSecond();
    add_source(u, u0);
    iElaps = cpuSecond();
    timeSource[z] = iElaps - iStart;
    //printf("add_source time %f\n", iElaps - iStart);
    add_source(v, v0);

    float alpha = DT * visc * N * N;
    float beta = 1 + 4 * alpha;
    SWAP(u0, u);

    iStart = cpuSecond();
    diffuse(1, u, u0, alpha, beta);
    iElaps = cpuSecond();
    timeDiffusion[z] = iElaps - iStart;
    //printf("diffuse time %f\n", iElaps - iStart);

    SWAP(v0, v);
    diffuse(2, v, v0, alpha, beta);

    iStart = cpuSecond();
    computeDivergenceAndPressure(u, v, u0, v0);
    iElaps = cpuSecond();
    timeDivergence[z] = iElaps - iStart;
    //printf("computeDivergenceAndPressure time %f\n", iElaps - iStart);

    alpha = 1;
    beta = 4;
    diffuse(0, u0, v0, alpha, beta);

    iStart = cpuSecond();
    lastProject(u, v, u0, v0);
    iElaps = cpuSecond();
    timeProjection[z] = iElaps - iStart;
    //printf("lastProject time %f\n", iElaps - iStart);

    SWAP(u0, u);
    SWAP(v0, v);

    iStart = cpuSecond();
    advect(1, u, u0, u0, v0);
    iElaps = cpuSecond();
    timeAdvection[z] = iElaps - iStart;
    //printf("advect time %f\n", iElaps - iStart);

    advect(2, v, v0, u0, v0);
    computeDivergenceAndPressure(u, v, u0, v0);
    diffuse(0, u0, v0, alpha, beta);
    lastProject(u, v, u0, v0);
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
    double sumElaps = 0, sumAdvection = 0, sumDiffusion = 0, sumDivergence = 0, sumProjection = 0, sumSource = 0;

    float *u = (float*)malloc(sizeof(float) * size);
    float *u_prev = (float*)malloc(sizeof(float) * size);
    float *v = (float*)malloc(sizeof(float) * size);
    float *v_prev = (float*)malloc(sizeof(float) * size);
    float *dens = (float*)malloc(sizeof(float) * size);
    float *dens_prev = (float*)malloc(sizeof(float) * size);

    double iStart, iElaps;

    int z = 0;
    int first = 1;
    //iStart = cpuSecond();
    while (z < Z) {
        iStart = cpuSecond();
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

        vel_step(u, v, u_prev, v_prev, VIS, z);
        dens_step(dens, dens_prev, u, v, DIFF);

        //printStateGrid(dens, u, v);
        //checkStability(u, v);
        iElaps = cpuSecond() - iStart;
        arrayElaps[z++] = iElaps;
    }
    //iElaps = cpuSecond() - iStart;
    for (int i = 0; i < Z; i++) {
        sumElaps += arrayElaps[i];
        sumSource += timeSource[i];
        sumDiffusion += timeDiffusion[i];
        sumDivergence += timeDivergence[i];
        sumAdvection += timeAdvection[i];
        sumProjection += timeProjection[i];
    }

    printf("Tot %f\nSource %f\nDiffusion %f\nDivergence %f\nAdvection %f\nProjection %f\n",
        sumElaps / Z, sumSource / Z, sumDiffusion / Z / 40, sumDivergence / Z, sumAdvection / Z, sumProjection / Z);

    //printStateGrid(dens, u, v);

    // Cleaning
    free(u); free(u_prev);
    free(v); free(v_prev);
    free(dens); free(dens_prev);

    return 0;
}

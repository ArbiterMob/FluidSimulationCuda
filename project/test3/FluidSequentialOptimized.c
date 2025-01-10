#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 126         // (WidthGrid - 2) 
#define DT 0.016f   // Instantaneous change in time (timestep)
#define VIS 0.0025f // Viscosity coefficient
#define DIFF 0.1f   // Diffusion coefficient

// SWAP macro
#define SWAP(x0, x) {float *tmp = x0; x0 = x; x = tmp;}

// Function to debug a float grid
void printDebug(char *string, float *x) {
    int i, j;

    printf("Debugging -> %s\n", string);

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
void add_sourceDens(float *x, float *s) {
    int i, size = (N + 2) * (N + 2);
    for (i = 0; i < size; i++)
        x[i] += DT * s[i];
}

void add_sourceVel(float *u, float *v, float *u0, float *v0) {
    int i, size = (N + 2) * (N + 2);
    for (i = 0; i < size; i++) {
        u[i] += DT * u0[i];
        v[i] += DT * v0[i];
    }
}

// Function to perform DIFFUSION for DENSITY and PRESSURE (using Jacobi iteration)
void diffuseDens (float *x, float *x0, float alpha, float beta) {
    int i, j, k, size = (N + 2) * (N + 2);
    //float a = DT * diff * N * N; // coefficient for the jacobi solution
    float *x_new = (float*)malloc(sizeof(float) * size); // temp grid to update values 
                                                         // after every jacobi iteration

    for (k = 0; k < 40; k++) {
        for (i = 1; i <= N; i++) {
            for (j = 1; j <= N; j++) {
                x_new[j + i * (N + 2)] = (x0[j + i * (N + 2)] + alpha * (x[(j - 1) + i * (N + 2)] + x[(j + 1) + i * (N + 2)] + x[j + (i - 1) * (N + 2)] +
                    x[j + (i + 1) * (N + 2)])) / beta;
            }
        }
        SWAP(x, x_new); // update the grid
        set_bnd(0, x);
    }
    free(x_new);
}

// Function to perform DIFFUSION for VELOCITY (using Jacobi iteration)
void diffuseVel (float *u, float *u0, float *v, float *v0, float alpha, float beta) {
    int i, j, k, size = (N + 2) * (N + 2);
    //float a = DT * diff * N * N; // coefficient for the jacobi solution
    float *u_new = (float*)malloc(sizeof(float) * size); // temp grid to update values 
                                                         // after every jacobi iteration
    float *v_new = (float*)malloc(sizeof(float) * size); // temp grid to update values 
                                                         // after every jacobi iteration

    for (k = 0; k < 40; k++) {
        for (i = 1; i <= N; i++) {
            for (j = 1; j <= N; j++) {
                u_new[j + i * (N + 2)] = (u0[j + i * (N + 2)] + alpha * (u[(j - 1) + i * (N + 2)] + u[(j + 1) + i * (N + 2)] + u[j + (i - 1) * (N + 2)] +
                    u[j + (i + 1) * (N + 2)])) / beta;
                v_new[j + i * (N + 2)] = (v0[j + i * (N + 2)] + alpha * (v[(j - 1) + i * (N + 2)] + v[(j + 1) + i * (N + 2)] + v[j + (i - 1) * (N + 2)] +
                    v[j + (i + 1) * (N + 2)])) / beta;
            }
        }
        SWAP(u, u_new); // update the grid
        SWAP(v, v_new); // update the grid
        set_bnd(1, u);
        set_bnd(2, v);
    }
    free(u_new);
    free(v_new);
}

// Function to perform ADVECTION for DENSITY (using bilinear interpolation)
void advectDens (float *d, float *d0, float *u, float *v) {
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
    set_bnd(0, d);
}

// Function to perform ADVECTION for VELOCITY (using bilinear interpolation)
void advectVel (float *u, float *v, float *u0, float *v0) {
    int i, j, i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    dt0 = DT * N; // N is the scale factor
    for (i = 1; i <= N; i++) {
        for (j = 1; j <= N; j++) {
            x = j - dt0 * u0[j + i * (N + 2)]; 
            y = i - dt0 * v0[j + i * (N + 2)];

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

            u[j + i * (N + 2)] = s0 * (t0 * u0[j0 + i0 * (N + 2)] + t1 * u0[j0 + i1 * (N + 2)]) +
                s1 * (t0 * u0[j1 + i0 * (N + 2)] + t1 * u0[j1 + i1 * (N + 2)]);
            v[j + i * (N + 2)] = s0 * (t0 * v0[j0 + i0 * (N + 2)] + t1 * v0[j0 + i1 * (N + 2)]) +
                s1 * (t0 * v0[j1 + i0 * (N + 2)] + t1 * v0[j1 + i1 * (N + 2)]);
        } 
    }
    set_bnd(1, u);
    set_bnd(2, v);
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
    diffuseDens(p, div, alpha, beta);

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
    add_sourceDens(x, x0);

    float alpha = DT * diff * N * N;
    float beta = 1 + 4 * alpha;
    SWAP(x0, x); 
    diffuseDens(x, x0, alpha, beta);

    SWAP(x0, x); 
    advectDens(x, x0, u, v);
}

// Function to simulate the evolution of velocity
void vel_step(float *u, float *v, float *u0, float *v0, float visc) {
    add_sourceVel(u, v, u0, v0);

    float alpha = DT * visc * N * N;
    float beta = 1 + 4 * alpha;
    SWAP(u0, u);
    SWAP(v0, v);
    diffuseVel(u, u0, v, v0, alpha, beta);
    project(u, v, u0, v0);

    SWAP(u0, u);
    SWAP(v0, v);
    advectVel(u, v, u0, v0);
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

    float *u = (float*)malloc(sizeof(float) * size);
    float *u_prev = (float*)malloc(sizeof(float) * size);
    float *v = (float*)malloc(sizeof(float) * size);
    float *v_prev = (float*)malloc(sizeof(float) * size);
    float *dens = (float*)malloc(sizeof(float) * size);
    float *dens_prev = (float*)malloc(sizeof(float) * size);

    double iStart, iElaps;

    int z = 0;
    int first = 1;
    iStart = cpuSecond();
    while (z++ < 5) {
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
    iElaps = cpuSecond() - iStart;
    printf("elapsed %f sec\n", iElaps);

    printStateGrid(dens, u, v);

    // Cleaning
    free(u); free(u_prev);
    free(v); free(v_prev);
    free(dens); free(dens_prev);

    return 0;
}
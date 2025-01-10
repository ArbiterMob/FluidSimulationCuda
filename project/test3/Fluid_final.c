#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N 6
#define DT 0.016f
#define VIS 0.0025f
#define DIFF 0.1f

#define IX(x, y) ((x) + (y) * (N + 2))
#define SWAP(x0, x) {float *tmp = x0; x0 = x; x = tmp;}

void printDebug(char *string, float *x);

// Function to measure time in seconds
double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void set_bnd(int b, float *x) {
    int i;

    for (i = 1; i <= N; i++) {
        x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
        x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
    }
    x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0 ,N+1)] = 0.5*(x[IX(1,N+1)]+x[IX(0 ,N )]);
    x[IX(N+1,0 )] = 0.5*(x[IX(N,0 )]+x[IX(N+1,1)]);
    x[IX(N+1,N+1)] = 0.5*(x[IX(N,N+1)]+x[IX(N+1,N )]);
}

void add_source(float *x, float *s, float dt) {
    int i, size = (N + 2) * (N + 2);
    for (i = 0; i < size; i++)
        x[i] += dt * s[i];
}

// GAUSS-SEIDEL
/*
void diffuse (int b, float *x, float *x0, float diff, float dt) {
    int i, j, k;
    float a = dt * diff * N * N;

    for (k = 0; k < 20; k++) {
        for (j = 1; j <= N; j++) {
            for (i = 1; i <= N; i++) {
                x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                    x[IX(i, j - 1)] + x[IX(i, j + 1)])) / (1 + 4 * a);
            }
        }
        set_bnd(b, x);
    }
}
*/

// JACOBI
void diffuse (int b, float *x, float *x0, float diff, float dt) {
    int i, j, k, size = (N + 2) * (N + 2);
    float a = dt * diff * N * N;
    float *x_new = (float*)malloc(sizeof(float) * size);
    printf("alpha is %f\n", a);

    for (k = 0; k < 40; k++) {
        for (j = 1; j <= N; j++) {
            for (i = 1; i <= N; i++) {
                x_new[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                    x[IX(i, j - 1)] + x[IX(i, j + 1)])) / (1 + 4 * a);
                //printf("x_new id %f in (%d, %d): x0[tid] %f | x[tid - 1] %f | d_x[tid + 1] %f, d_x[tid - N - 2] %f | d_x[tid + N + 2] %f | beta %f\n",
                //    x_new[IX(i, j)], i, j, x0[IX(i, j)], x[IX(i - 1, j)],  x[IX(i + 1, j)], x[IX(i, j - 1)], x[IX(i, j + 1)], 1 + 4 * a);
            }
        }
        
        /*
        for (j = 1; j <= N; j++) {
            for (i = 1; i <= N; i++) {
                x[IX(i, j)] = x_new[IX(i, j)];
            }
        }
        */
        SWAP(x, x_new);

        set_bnd(b, x);

        //printf("DIFFUSION- k %d\n", k);
        //printDebug("", x);
    }

    free(x_new);
}

void advect (int b, float *d, float *d0, float *u, float *v, float dt) {
    int i, j, i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    dt0 = dt * N;
    for (j = 1; j <= N; j++) {
        for (i = 1; i <= N; i++) {
            x = i - dt0 * u[IX(i, j)]; 
            y = j - dt0 * v[IX(i, j)];

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

            d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) + 
                s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        } 
    }
    set_bnd(b, d);
}

// PROJECTION WITH GAUSS-SEIDEL
/*
void project(float *u, float *v, float *p, float *div) {
    int i, j, k, size = (N + 2) * (N + 2);
    float h;

    h = 1.0 / N;
    for (j = 1; j <= N; j++) {
        for (i = 1; i <= N; i++) {
            div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + 
                v[IX(i, j + 1)] + v[IX(i, j - 1)]);
            p[IX(i, j)] = 0;
        }
    }
    set_bnd(0, div);
    set_bnd(0, p);

    for (k = 0; k < 40; k++) {
        for (j = 1; j <= N; j++) {
            for (i = 1; i <= N; i++) {
                p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                    p[IX(i, j - 1)] + p[IX(i, j + 1)]) / 4;
            }
        }

        set_bnd(0, p);
    }

    for (j = 1; j <= N; j++) {
        for (i = 1; i <= N; i++) {
            u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
            v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;


        }
    }
    set_bnd(1, u);
    set_bnd(2, v);
}
*/

// PROJECTION WITH JACOBI
void project(float *u, float *v, float *p, float *div) {
    int i, j, k, size = (N + 2) * (N + 2);
    float h;
    float *p_new = (float*)malloc(sizeof(float) * size);

    h = 1.0 / N;
    for (j = 1; j <= N; j++) {
        for (i = 1; i <= N; i++) {
            div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + 
                v[IX(i, j + 1)] - v[IX(i, j - 1)]);
            p[IX(i, j)] = 0;
            //printf("div[tid] %f (%d, %d)| u[tid + 1] %f | u[tid - 1] %f | v[tid + N + 2] %f | v[tid - N - 2] %f\n",
            //div[IX(i, j)], i, j, u[IX(i + 1, j)], u[IX(i - 1, j)], v[IX(i, j + 1)], v[IX(i, j - 1)]);
        }
    }
    set_bnd(0, div);
    set_bnd(0, p);

    //printDebug("Divergence div\n", div);

    for (k = 0; k < 40; k++) {
        for (j = 1; j <= N; j++) {
            for (i = 1; i <= N; i++) {
                p_new[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                    p[IX(i, j - 1)] + p[IX(i, j + 1)]) / 4;
            }
        }

        /*
        for (j = 1; j <= N; j++) {
            for (i = 1; i <= N; i++) {
                p[IX(i, j)] = p_new[IX(i, j)];
            }
        }
        */
        SWAP(p, p_new);

        set_bnd(0, p);
    }

    for (j = 1; j <= N; j++) {
        for (i = 1; i <= N; i++) {
            u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
            v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;


        }
    }
    set_bnd(1, u);
    set_bnd(2, v);
}

void dens_step(float *x, float *x0, float *u, float *v, float diff, float dt, int first) {
    add_source(x, x0, dt);

    SWAP(x0, x); 
    diffuse(0, x, x0, diff, dt);

    SWAP(x0, x); 
    advect(0, x, x0, u, v, dt);
}

/*
void vel_step(float *u, float *v, float *u0, float *v0, float visc, float dt, int first) {
    printf("BEFORE ADDING SOURCE START\n");
    printDebug("u_prev", u0);
    printDebug("u", u);
    printDebug("v_prev", v0);
    printDebug("v", v);
    printf("BEFORE ADDING SOURCE END\n");

    printf("ADDING SOURCE START\n");
    add_source(u, u0, dt);
    printDebug("u_prev", u0);
    printDebug("u", u);
    add_source(v, v0, dt);
    printDebug("v_prev", v0);
    printDebug("v", v);
    printf("ADDING SOURCE END\n");

    // for the diffusion step we need to copy ...

    printf("AFTER SWAP DIFFUSE u START\n");
    SWAP(u0, u);
    printDebug("u_prev", u0);
    printDebug("u", u);
    printf("AFTER SWAP DIFFUSE u END\n");

    printf("DIFFUSE START u\n");
    diffuse(1, u, u0, visc, dt);
    printDebug("u_prev", u0);
    printDebug("u", u);
    printf("DIFFUSE END u\n");

    printf("AFTER SWAP DIFFUSE v START\n");
    SWAP(v0, v);
    printDebug("v_prev", v0);
    printDebug("v", v);
    printf("AFTER SWAP DIFFUSE v END\n");

    printf("DIFFUSE START v\n");
    diffuse(1, v, v0,    diffuse(1, u, u0, visc, dt);
 visc, dt);
    printDebug("v_prev", v0);
    printDebug("v", v);
    printf("DIFFUSE END v\n");

    printf("FIRST PROJECTION START\n");
    project(u, v, u0, v0);
    printDebug("u_prev", u0);
    printDebug("u", u);
    printDebug("v_prev", v0);
    printDebug("v", v);
    printf("FIRST PROJECTION END\n");

    SWAP(u0, u);
    SWAP(v0, v);
    printf("AFTER SECOND SWAP BEFORE ADVECTION START\n");
    printDebug("u_prev", u0);
    printDebug("u", u);
    printDebug("v_prev", v0);
    printDebug("v", v);
    printf("AFTER SECOND SWAP BEFORE ADVECTION END\n");

    advect(1, u, u0, u0, v0, dt);
    advect(2, v, v0, u0, v0, dt);
    printf("AFTER ADVECTION START\n");
    printDebug("u_prev", u0);
    printDebug("u", u);
    printDebug("v_prev", v0);
    printDebug("v", v);
    printf("AFTER ADVECTION END\n");

    project(u, v, u0, v0);
    printf("LAST PROJECTION START\n");
    printDebug("u_prev", u0);
    printDebug("u", u);
    printDebug("v_prev", v0);
    printDebug("v", v);
    printf("LAST PROJECTION END\n");
}
*/

void vel_step(float *u, float *v, float *u0, float *v0, float visc, float dt, int first) {
    add_source(u, u0, dt);
    add_source(v, v0, dt);

    // for the diffusion step we need to copy ... ???

    SWAP(u0, u);
    //printf("AFTER SOURCE u\n");
    //printDebug("", u0);
    diffuse(1, u, u0, visc, dt);
    SWAP(v0, v);
    //printf("AFTER SOURCE v\n");
    //printDebug("", v0);
    diffuse(2, v, v0, visc, dt);
    //printf("AFTER DIFFUSION - STARTING PROJECTION\n");
    project(u, v, u0, v0);

    SWAP(u0, u);
    SWAP(v0, v);
    advect(1, u, u0, u0, v0, dt);
    advect(2, v, v0, u0, v0, dt);
    project(u, v, u0, v0);

    //printf("AFTER LAST PROJECTION\n");
    //printDebug("", u);
    //printDebug("", v);
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

void printSomething(float *dens, float *u, float *v) {
    int i, j;

    printf("---------------------------------------\n");
    printf("DENSITY\n");
    for (i = 0; i < N + 2; i++) {
        for (j = 0; j < N + 2; j++) {
            printf("[%f] ", dens[IX(j, i)]);
        }
        printf("\n");
    }
    printf("\n\n");

    printf("VELOCITY\n");
    for (i = 0; i < N + 2; i++) {
        for (j = 0; j < N + 2; j++) {
            printf("[%f, %f] ", u[IX(j, i)], v[IX(j, i)]);
        }
        printf("\n");
    }
}

void printDebug(char *string, float *x) {
    int i, j;

    printf("Debugging -> %s\n", string);

    for (j = 0; j < N + 2; j++) {
        for (i = 0; i < N + 2; i++) {
            printf("[%f] ", x[IX(i, j)]);
        }
        printf("\n");
    }
    printf("\n\n");
}

int checkStability(float *u, float *v) {
    int i, j, h = 1.0 / N;
    float error = 1.0E-8;
    float found = 0;
    for (i = 0; i < N + 2; i++) {
        for (j = 0; j < N + 2; j++) {
            float divergence = 0.5 * h * ((u[IX(j + 1, i)] - u[IX(j - 1, i)]) + 
                v[IX(j, i + 1)] - v[IX(j, i - 1)]);
            if (divergence > error) {
                printf("The velocity is not divergence-free\n");
                found = 1;
            }
        }
    }

    found ? printf("all bad\n") : printf("all good\n");

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
            //printSomething(dens_prev, u_prev, v_prev);
            //printf("Hello end\n");
        } else {
            for (int i = 0; i < size; i++) {
                u_prev[i] = 0.0f;
                v_prev[i] = 0.0f;
                dens_prev[i] = 0.0f;
            }
        }


        vel_step(u, v, u_prev, v_prev, VIS, DT, first);
        dens_step(dens, dens_prev, u, v, DIFF, DT, first);



        //printSomething(dens, u, v);
        //checkStability(u, v);
    }
    iElaps = cpuSecond() - iStart;
    printf("elapsed %f sec\n", iElaps);

    printSomething(dens, u, v);


    // Cleaning
    free(u); free(u_prev);
    free(v); free(v_prev);
    free(dens); free(dens_prev);

    return 0;
}
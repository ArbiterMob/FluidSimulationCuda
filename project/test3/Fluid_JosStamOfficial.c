/*
THE DENSITY STILL DIVERGES !!!
-> maybe we don't use it ...
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 8
#define DT 0.016f
#define VIS 0.0025f
#define DIFF 0.1f

#define IX(i, j) ((i) + (j) * (N + 2))
#define SWAP(x0, x) {double *tmp = x0; x0 = x; x = tmp;}

void set_bnd(int b, double *x) {
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

void add_source(double *x, double *s, double dt) {
    int i, size = (N + 2) * (N + 2);
    for (i = 0; i < size; i++)
        x[i] += dt * s[i];
}

void diffuse(int b, double *x, double *x0, double diff, double dt) {
    int i, j, k;
    double a = dt * diff * N * N; // still doesn't convince me ...

    for (k = 0; k < 20; k++) {
        for (i = 1; i <= N; i++) {
            for (j = 1; j <= N; j++) {
                x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                    x[IX(i, j - 1)] + x[IX(i, j + 1)])) / (1 + 4 * a);
            }
        }
        set_bnd(b, x);
    }
}

void advect(int b, double *d, double *d0, double *u, double *v, double dt) {
    int i, j, i0, j0, i1, j1;
    double x, y, s0, t0, s1, t1, dt0;

    dt0 = dt * N;
    for (i = 1; i <= N; i++) {
        for (j = 1; j <= N; j++) {
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

            d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)] + 
                s1 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        } 
    }
    set_bnd(b, d);
}

void project(double *u, double *v, double *p, double *div) {
    int i, j, k;
    double h;

    h = 1.0 / N;
    for (i = 1; i <= N; i++) {
        for (j = 1; j <= N; j++) {
            // I don't know why it multiplies by h. Shouldn't it divide by h ????
            div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + 
                v[IX(i, j + 1)] + v[IX(i, j - 1)]);
            //div[IX(i, j)] = -0.5 * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + 
            //    v[IX(i, j + 1)] + v[IX(i, j - 1)]) / h;
            p[IX(i, j)] = 0;
        }
    }
    set_bnd(0, div);
    set_bnd(0, p);

    for (k = 0; k < 20; k++) {
        for (i = 1; i <= N; i++) {
            for (j = 1; j <= N; j++) {
                p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                    p[IX(i, j - 1)] + p[IX(i, j + 1)]) / 4;
            }
        }
        set_bnd(0, p);
    }

    for (i = 1; i <= N; i++) {
        for (j = 1; j <= N; j++) {
            // he divides by h (not multiply) -> correct ?, but why in the divergence we did the inverse????
            //u[IX(i, j)] -= 0.5 * h * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
            //v[IX(i, j)] -= 0.5 * h * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);

            u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
            v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;


        }
    }
    set_bnd(1, u);
    set_bnd(2, v);
}

void dens_step(double *x, double *x0, double *u, double *v, double diff, double dt) {
    add_source(x, x0, dt);
    SWAP(x0, x); diffuse(0, x, x0, diff, dt);
    SWAP(x0, x); advect(0, x, x0, u, v, dt);
}

void vel_step(double *u, double *v, double *u0, double *v0, double visc, double dt) {
    add_source(u, u0, dt); add_source(v, v0, dt);
    SWAP(u0, u); diffuse(1, u, u0, visc, dt);
    SWAP(v0, v); diffuse(2, v, v0, visc, dt);
    project(u, v, u0, v0);
    SWAP(u0, u); SWAP(v0, v);
    advect(1, u, u0, u0, v0, dt); advect(2, v, v0, u0, v0, dt);
    project(u, v, u0, v0);
}

void initializeParameters(double *dens, double *dens_prev, double *u, double *u_prev, double *v, double *v_prev) {
    int i, j;
    int center_x = (N + 2) / 2, center_y = (N + 2) / 2;
    int radius = (N + 2) / 8;

    // density 
    for (i = 0; i < N + 2; i ++) {
        for (j = 0; j < N + 2; j++) {
            double dist = sqrt((j - center_x) * (j - center_x) + (i - center_y) * (i - center_y));
            if (dist < radius) 
                dens_prev[IX(j, i)] = (1.0f - (dist / radius));
            else 
                dens_prev[IX(j, i)] = 0.0f;
            
            dens[IX(j, i)] = 0.0f;
        }
    } 

    // velocity
    for (i = 0; i < N + 2; i ++) {
        for (j = 0; j < N + 2; j++) {
            u_prev[IX(j, i)] = ((rand() % 100) / 100.0f) * 0.1f; // Random values between 1.0 and 10.0
            v_prev[IX(j, i)] = ((rand() % 100) / 100.0f) * 0.1f;// Random values between 1.0 and 10.0

            u[IX(j, i)] = 0.0f;
            v[IX(j, i)] = 0.0f;
        }
    }

}

void printSomething(double *dens, double *u, double *v) {
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

int checkStability(double *u, double *v) {
    int i, j, h = 1.0 / N;
    double error = 1.0E-8;
    double found = 0;
    for (i = 0; i < N + 2; i++) {
        for (j = 0; j < N + 2; j++) {
            double divergence = 0.5 * h * ((u[IX(j + 1, i)] - u[IX(j - 1, i)]) + 
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

    double *u = (double*)malloc(sizeof(double) * size);
    double *u_prev = (double*)malloc(sizeof(double) * size);
    double *v = (double*)malloc(sizeof(double) * size);
    double *v_prev = (double*)malloc(sizeof(double) * size);
    double *dens = (double*)malloc(sizeof(double) * size);
    double *dens_prev = (double*)malloc(sizeof(double) * size);

    int z = 0;
    int first = 1;
    while (z++ < 1000) {
        if (first) {
            initializeParameters(dens, dens_prev, u, u_prev, v, v_prev);
            first = 0;
            printf("HELLO init\n");
            printSomething(dens_prev, u_prev, v_prev);
            printf("Hello end\n");
        }

        vel_step(u, v, u_prev, v_prev, VIS, DT);
        dens_step(dens, dens_prev, u, v, DIFF, DT);

        printSomething(dens, u, v);
        checkStability(u, v);
    }

    // Cleaning
    free(u); free(u_prev);
    free(v); free(v_prev);
    free(dens); free(dens_prev);

    return 0;
}
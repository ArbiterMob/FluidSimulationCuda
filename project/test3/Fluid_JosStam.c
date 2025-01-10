/*
 * If you want to show the results of the simulation, you have several choices. 
 * One would be to rewrite the simulator in two dimensions instead of three, 
 * and you can then just display the density pointer onscreen as a regular image.
 * Another choice would be to run the full 3D simulation but only display 2D slice of it. 
 * And finally, another choice would be to run a simulation small enough that you can just
 * draw one polygon per cell without taking too much of a speed hit.
*/

/*
 * In a typical implementation, the various components are not computed and added together,
 * as in Equation 11. Instead, the solution is found via composition of transformations 
 * on the state; in other words, each component is a step that takes a field as input,
 * and produces a new field as output. We can define an operator s.jpg that is equivalent 
 * to the solution of Equation 11 over a single time step. The operator is defined as the
 * composition of operators for advection ( a.jpg ), diffusion ( d.jpg ), 
 * force application ( f.jpg ), and projection ( p.jpg ). 
*/

/*
 * Because texture coordinates are not in the same units as our simulation domain (the
 * texture coordinates are in the range [0, N], where N is the grid resolution), we must
 * scale the velocity into grid space. This is reflected in Cg code with the multiplication
 * of the local velocity by the parameter rdx, which represents the reciprocal of the grid scale x.
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <GLFW/glfw3.h>
#include <math.h>

#define N 100               // Dimension of the grid
#define DT 0.00016f            // Delta T for interative solver
#define VIS 0.0025f         // Viscosity constant
#define FORCE (5.8f * DIM)  // Force scale factor
#define DIFF 0.1f           // Density of the dye
#define SCALE 20.0f

#define IX(x, y) ((x) + (y) * (N + 2))
#define SWAP(x0, x) {float *tmp = x0; x0 = x; x = tmp;}

/*
We assume that the fluid is contained in a box with solid walls: no flow should exit 
the walls. This simply means that the horizontal component of the velocity should be
zero on the vertical walls, while the vertical component of the veloicty should be 
zero on the horizontal walls. For the density and other field considered in the code
we simply assume continuity.
*/
void set_bnd(int b, float *x){
    int i;

    for (i = 1; i <= N; i++) { 
        x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
        x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
    }

    // corners
    x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, N + 1)] = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
    x[IX(N + 1, 0)] = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
    x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

void add_source(float *x, float *s, float dt) {
    int i, size = (N + 2) * (N + 2);
    for (i = 0; i < size; i++)
        x[i] += dt * s[i];
}

/*
For large diffusion rates a, the density values start to oscillate, become negative and 
finally diverge, making the simulation useless. This behavior is a general problem that
plagues unstable methods.

void diffuse_bad (int b, float *x, float *x0, float diff, float dt) {
    int i, j;
    float a = dt * diff * N * N; // what is this ???

    for (i = 1; i <= N; i++) {
        for (j = 1; j <= N; j++) {
            x[IX(j, i)] = x0[IX(j, i)] + a * (x0[IX(j - 1, i)] + x0[IX(j + 1, i)] +
                x0[IX(j, i - 1)] + x0[IX(j, i + 1)] - 4 * x0[IX(j, i)]);
        }
    }

    set_bnd(b, x);
}
*/

/*
We consider a stable method for the diffusion step. The basic idea behind our method is to
find the densities which when diffused backward in time yield the densities we started with. 
In code:

x0[IX(j, i)] = x[IX(j, i)] - a * (x[IX(j - 1, i)] + x[IX(j + 1, i)] + x[IX(j, i + 1)] 
    -4 * x[IX(j, i)]);
*/

// Gauss-Seidel relaxation - MAYBE JACOBI -> look into it
// okay, this is the jacobi iteration ...
void diffuse(int b, float *x, float *x0, float diff, float dt) {
    int i, j, k;
    float a = dt * diff * N * N; // again, I don't know what this is
                                 // maybe this is the alpha in my code ...
                                 // IT IS the alpha in the code

    for (k = 0; k < 20; k++) {
        for (i = 1; i <= N; i++) {
            for (j = 1; j <= N; j++) {
                x[IX(j, i)] = (x0[IX(j, i)] + a * (x[IX(j - 1, i)] + x[IX(j + 1, i)] +
                    x[IX(j, i - 1)] + x[IX(j, i + 1)])) / (1 + 4 * a);
            }
        }
    }

    set_bnd(b, x);
}

/*
Find the particles which over a single time step end up exactly at the grid cell's centers.
The amount of density that these particles carry is simply obtained by linearly interpolating
the density at theri starting location from the four closest neighbours.

This suggests the following update procedure for the density. Start with two grids: one that
contains the density values from the previous time step adn one that will contain the new values.
For each grid cell of the latter we trace the cell's center position backwards through the
velocity field.
*/
void advect(int b, float *d, float *d0, float *u, float *v, float dt) {
    int i, j, i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    dt0 = dt * N; // similar to timestep * rdx ??
    for (i = 1; i <= N; i++) {
        for (j = 1; j <= N; j++) {
            x = j - dt0 * u[IX(j, i)];
            y = i - dt0 * v[IX(j, i)];

            // bilinear interpolation
            if (x < 0.5) x = 0.5;
            if (x > N + 0.5) x = N + 0.5;
            j0 = (int)x;
            j1 = j0 + 1;

            if (y < 0.5) y = 0.5;
            if (y > N + 0.5) y = N + 0.5;
            i0 = (int)y;
            i1 = i0 + 1;

            s1 = x - j0;
            s0 = 1 - s1;
            t1 = y - i0;
            t0 = 1 - t1;

            d[IX(j, i)] = s0 * (t0 * d0[IX(j0, i0)] + t1 * d0[IX(j0, i1)]) +
                s1 * (t0 * d0[IX(j1, i0)] + t1 * d0[IX(j1, i1)]);
        }
    }

    set_bnd(b, d);
}

/*
This routine forces the velocity to be mass conserving (divergence-free). This is an 
important property of real fluids which should be enforced. -> Hodge decomposition 
-> solving Poisson equation. This system is sparse and we ca re-use our Gauss-Seidel relaxation
code developed for the density diffusion step to solve it.
*/
// Where is the pressure and the divergence ??? -> they are computed in the routine
void project(float *u, float *v, float *p, float *div) {
    int i, j, k;
    float h;

    // divergence and pressure
    h = 1.0 / N;
    for (i = 1; i <= N; i++) {
        for (j = 1; j <= N; j++) {
            div[IX(j, i)] = -0.5 * h * (u[IX(j + 1, i)] - u[IX(j - 1, i)] 
                + v[IX(j, i + 1)] - v[IX(j, i - 1)]);
            p[IX(j, i)] = 0;
        }
    }
    set_bnd(0, div);
    set_bnd(0, p);

    // Jacobi iteration to find p
    for (k = 0; k < 20; k++) {
        for (i = 1; i <= N; i++) {
            for (j = 1; j <= N; j++) {
                p[IX(j, i)] = (div[IX(j, i)] + p[IX(j - 1, i)] + p[IX(j + 1, i)] +
                    p[IX(j, i - 1)] + p[IX(j, i + 1)]) / 4;
            }
        }
    }
    set_bnd(0, p);

    // gradient of p and new velocity
    for (i = 1; i <= N; i++) {
        for (j = 1; j <= N; j++) {
            u[IX(j, i)] -= 0.5 * (p[IX(j + 1, i)] - p[IX(j - 1, i)]) / h;
            v[IX(j, i)] -= 0.5 * (p[IX(j, i + 1)] - p[IX(j, i - 1)]) / h;
        }
    }
    set_bnd(1, u);
    set_bnd(2, v);
}

/*
Draw dens
*/
void draw_dens(float *dens) {
    int i, j;
    int h = 1.0f / N;

    glClear(GL_COLOR_BUFFER_BIT);
    
    glBegin(GL_QUADS);
    for (i = 0; i < N + 2; i++) {
        for (j = 0; j < N + 2; j++) {
            float value = dens[IX(j, i)];
            glColor3f(value, value, value); // Greyscale value;

            // Compute the square's center
            float squareX = j * SCALE;
            float squareY = i * SCALE;

            // Render a square centered at (squareX, squareY)
            float squareSize = SCALE * 0.4f; // Size of the square as a fraction of the scale
            glVertex2f(squareX - squareSize, squareY - squareSize); // Bottom-left
            glVertex2f(squareX + squareSize, squareY - squareSize); // Bottom-right
            glVertex2f(squareX + squareSize, squareY + squareSize); // Top-right
            glVertex2f(squareX - squareSize, squareY + squareSize); // Top-left
        }
    }
    glEnd();
}

/*
inizialization
*/
void initializeParameters(float *dens, float *dens_prev, float *u, float *u_prev, float *v, float *v_prev) {
    int i, j;
    int center_x = (N + 2) / 2, center_y = (N + 2) / 2;
    int radius = 1; //(N + 2) / 8;

    // density 
    for (i = 0; i < N + 2; i ++) {
        for (j = 0; j < N + 2; j++) {
            float dist = sqrt((j - center_x) * (j - center_x) + (i - center_y) * (i - center_y));
            if (dist < radius) 
                dens_prev[IX(j, i)] = (1.0f - (dist / radius)) * 1000;
            else 
                dens_prev[IX(j, i)] = 0.0f;
            
            dens[IX(j, i)] = 0.0f;
        }
    } 

    // velocity
    for (i = 0; i < N + 2; i ++) {
        for (j = 0; j < N + 2; j++) {
            u_prev[IX(j, i)] = ((rand() % 100) / 100.0f) * 0.1f; // Random values between 0.0 and 0.1
            v_prev[IX(j, i)] = ((rand() % 100) / 100.0f) * 0.1f; // Random values between 0.0 and 0.1

            u[IX(j, i)] = 0.0f;
            v[IX(j, i)] = 0.0f;
        }
    }

}

/*
Single routine for the density solver. We assume that the source densities are initially
contained in the x0 array.
*/
void dens_step(float *x, float *x0, float *u, float *v, float diff, float dt) {
    add_source(x, x0, dt);
    SWAP(x0, x); diffuse(0, x, x0, diff, dt);
    SWAP(x0, x); advect(0, x, x0, u, v, dt);
}

/*
Single routine for the velocity solver. We assume that the velocity field prev is stored in 
the arrays u0 and v0.
*/
void vel_step(float *u, float *v, float *u0, float *v0, float visc, float dt) {
    add_source(u, u0, dt);
    add_source(v, v0, dt);
    SWAP(u0, u); diffuse(1, u, u0, visc, dt);
    SWAP(v0, v); diffuse(2, v, v0, visc, dt);
    project(u, v, u0, v0);
    SWAP(u0, u); SWAP(v0, v);
    advect(1, u, u0, u0, v0, dt); advect(2, v, v0, u0, v0, dt);
    project(u, v, u0, v0);
}

/*
FUNCTIONs FOR DEBUGGING
*/
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

int main(int argc, char** argv) {
    int size = (N + 2) * (N + 2);
    //float h = 1.0f / N; // each side of the grid is one, so the grid spacing is given by h = 1 / N

    float *u = (float*)malloc(sizeof(float) * size); 
    float *v = (float*)malloc(sizeof(float) * size);
    float *u_prev = (float*)malloc(sizeof(float) * size); 
    float *v_prev = (float*)malloc(sizeof(float) * size);

    float *dens = (float*)malloc(sizeof(float) * size);
    float *dens_prev = (float*)malloc(sizeof(float) * size);

    // Graphics Inizialization
    /*
    glfwInit();
    GLFWwindow *window = glfwCreateWindow(N + 2, N + 2, "Fluid Simulation", NULL, NULL);
    glfwMakeContextCurrent(window);
    glViewport(0, 0, (N + 2) * SCALE, (N + 2) * SCALE);
    glOrtho(0, (N + 2) * SCALE, 0, (N + 2) * SCALE, -1, 1);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    */

    initializeParameters(dens, dens_prev, u, u_prev, v, v_prev);
    //printSomething(dens, u, v);

    int z = 0;
    int first = 1;
    while(z++ < 1000) {
    //while (!glfwWindowShouldClose(window)) {
        //get_from_UI(dens_prev, u_prev, v_prev);
        
        if (first) { 
            initializeParameters(dens, dens_prev, u, u_prev, v, v_prev); 
            first = 0;
        }
        /*
        else {
            int size = (N + 2) * (N + 2);
            for (int i = 1; i < size; i++) {
                dens_prev[i] = 0.0f;
                u_prev[i] = 0.0f;
                v_prev[i] = 0.0f;
            }
        }
        */

        vel_step(u, v, u_prev, v_prev, VIS, DT);
        dens_step(dens, dens_prev, u, v, DIFF, DT);

        printSomething(dens, u, v);
        checkStability(u, v);

        // Render
        /*
        glClear(GL_COLOR_BUFFER_BIT);
        draw_dens(dens);
        glfwSwapBuffers(window);
        glfwPollEvents();
        */
    }

    // Cleaning
    free(u); free(u_prev);
    free(v); free(v_prev);
    free(dens); free(dens_prev);
    /*
    glfwDestroyWindow(window);
    glfwTerminate();
    */

    return 0;
}
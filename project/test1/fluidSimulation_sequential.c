#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <GLFW/glfw3.h>

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b)

const int W = 8;
const float TIMESTEP = 0.09f;
const int NUM_JACOBI_ITERATIONS = 20;
const float VISCOSITY = 0.0025f;
const float SCALE = 20.0f; // Visualization scaling factor

typedef struct {
    int x;
    int y;
} Point2D;

typedef struct {
    int width;
    int height;
    float *values;
} ScalarField2D;

typedef struct {
    float x;
    float y;
} Vector2D;

typedef struct {
    int width;
    int height;
    Vector2D *vectors;
} VectorField2D;

// UTILS
float computeMaxMagnitude(VectorField2D field) {
    float maxMagnitude = 0.0f;
    for (int i = 0; i < field.width * field.height; i++) {
        Vector2D v = field.vectors[i];
        float length = sqrtf(v.x * v.x + v.y * v.y);
        if (length > maxMagnitude) {
            maxMagnitude = length;
        }
    }
    return maxMagnitude;
}

// ADVECTION
void advectVectorField(Point2D coords, Vector2D* advectedQty, float timestep, float rdx, VectorField2D inputVelocity, VectorField2D qtyToAdvect) {
   // Follow the velocity field "back in time"
    Point2D pos;
    pos.x = coords.x - timestep * rdx * inputVelocity.vectors[coords.x + coords.y * inputVelocity.width].x;
    pos.y = coords.y - timestep * rdx * inputVelocity.vectors[coords.x + coords.y * inputVelocity.width].y;

    // Compute the integer and fractional parts of the position
    int x0 = (int)floor(pos.x);
    int x1 = x0 + 1;
    int y0 = (int)floor(pos.y);
    int y1 = y0 + 1;

    float sx = pos.x - x0; // Fractional part in x direction
    float sy = pos.y - y0; // Fractional part in y direction

    // Clamp the surrounding grid points individually
    x0 = MAX(0, MIN(x0, inputVelocity.width - 1));
    x1 = MAX(0, MIN(x1, inputVelocity.width - 1));
    y0 = MAX(0, MIN(y0, inputVelocity.height - 1));
    y1 = MAX(0, MIN(y1, inputVelocity.height - 1));

    // Retrieve the values at the four surrounding grid points
    Vector2D q00 = qtyToAdvect.vectors[x0 + y0 * qtyToAdvect.width];
    Vector2D q10 = qtyToAdvect.vectors[x1 + y0 * qtyToAdvect.width];
    Vector2D q01 = qtyToAdvect.vectors[x0 + y1 * qtyToAdvect.width];
    Vector2D q11 = qtyToAdvect.vectors[x1 + y1 * qtyToAdvect.width];

    // Bilinear interpolation
    Vector2D q0, q1, qInterpolated;
    q0.x = (1 - sx) * q00.x + sx * q10.x;
    q0.y = (1 - sx) * q00.y + sx * q10.y;

    q1.x = (1 - sx) * q01.x + sx * q11.x;
    q1.y = (1 - sx) * q01.y + sx * q11.y;

    qInterpolated.x = (1 - sy) * q0.x + sy * q1.x;
    qInterpolated.y = (1 - sy) * q0.y + sy * q1.y;

    // Write the interpolated value to the output
    *advectedQty = qInterpolated;
    //printf("advected quantity: [%f, %f]\n", advectedQty->x, advectedQty->y);
}

/*
void advectScalarField(Point2D coords, float* advectedQty, float timestep, float rdx, VectorField2D inputVelocity, ScalarField2D qtyToAdvect) {
    // Follow the velocity field "back in time"
    Point2D pos;
    pos.x = coords.x - timestep * rdx * inputVelocity.vectors[coords.x + coords.y * inputVelocity.width].x;
    pos.y = coords.y - timestep * rdx * inputVelocity.vectors[coords.x + coords.y * inputVelocity.width].y;

    pos.x = fmaxf(0.0f, fminf(pos.x, inputVelocity.width - 1));
    pos.y = fmaxf(0.0f, fminf(pos.y, inputVelocity.height - 1));

    // interpolare (adesso no per semplicità) and write the output
    *advectedQty = qtyToAdvect.values[pos.x + pos.y * qtyToAdvect.width];
}
*/


// VISCOUS DIFFUSION
void jacobiVectorField(Point2D coords, Vector2D* result, float alpha, float rBeta, VectorField2D inputVelocity, VectorField2D b) {
    Vector2D xL = {0, 0}, xR = {0, 0}, xB = {0, 0}, xT = {0, 0}, bC;

    // Bounds-checked sampling
    if (coords.x - 1 >= 0) xL = inputVelocity.vectors[(coords.x - 1) + coords.y * inputVelocity.width];
    if (coords.x + 1 < inputVelocity.width) xR = inputVelocity.vectors[(coords.x + 1) + coords.y * inputVelocity.width];
    if (coords.y + 1 < inputVelocity.height) xB = inputVelocity.vectors[coords.x + (coords.y + 1) * inputVelocity.width];
    if (coords.y - 1 >= 0) xT = inputVelocity.vectors[coords.x + (coords.y - 1) * inputVelocity.width];

    // Center sample
    bC = b.vectors[coords.x + coords.y * b.width]; 

    // Evaluate Jacobi iteration
    result->x = (xL.x + xR.x + xB.x + xT.x + alpha * bC.x) * rBeta;
    result->y = (xL.y + xR.y + xB.y + xT.y + alpha * bC.y) * rBeta;

    //printf("result of jacobi for velocity: [%f, %f]\n", result->x, result->y);
}   

void jacobiScalarField(Point2D coords, float* result, float alpha, float rBeta, ScalarField2D inputPressure, ScalarField2D b) {
    float xL = 0, xR = 0, xB = 0, xT = 0, bC;

    // Bounds-checked sampling
    if (coords.x - 1 >= 0) xL = inputPressure.values[(coords.x - 1) + coords.y * inputPressure.width];
    if (coords.x + 1 < inputPressure.width) xR = inputPressure.values[(coords.x + 1) + coords.y * inputPressure.width];
    if (coords.y + 1 < inputPressure.height) xB = inputPressure.values[coords.x + (coords.y + 1) * inputPressure.width];
    if (coords.y - 1 >= 0) xT = inputPressure.values[coords.x + (coords.y - 1) * inputPressure.width];

    // Center sample
    bC = b.values[coords.x + coords.y * b.width]; 

    // Evaluate Jacobi iteration
    *result = (xL + xR + xB + xT + alpha * bC) * rBeta;

    //printf("result of jacobi for pressure: [%f]\n", *result);
}   

// FORCE APPLICATION
void addExternalForce(Vector2D* velocity, float fx, float fy) {
    velocity->x += fx * TIMESTEP;
    velocity->y += fy * TIMESTEP;
}

// PROJECTION
void divergence(Point2D coords, float* div, float halfrdx, VectorField2D w) {
    Vector2D wL = {0,0}, wR = {0,0}, wB = {0,0}, wT = {0,0};

    if (coords.x - 1 >= 0) wL = w.vectors[(coords.x - 1) + coords.y * w.width];
    if (coords.x + 1 < w.width) wR = w.vectors[(coords.x + 1) + coords.y * w.width];
    if (coords.y - 1 >= 0) wB = w.vectors[coords.x + (coords.y - 1) * w.width];
    if (coords.y + 1 < w.height) wT = w.vectors[coords.x + (coords.y + 1) * w.width];

    *div = halfrdx * ((wR.x - wL.x) + (wT.y - wB.y));
}

void gradient(Point2D coords, Vector2D* uNew, float halfrdx, ScalarField2D p, VectorField2D w) {
    float pL = 0, pR = 0, pB = 0, pT = 0;

    if (coords.x - 1 >= 0) pL = p.values[(coords.x - 1) + coords.y * w.width];
    if (coords.x + 1 < p.width) pR = p.values[(coords.x + 1) + coords.y * w.width];
    if (coords.y - 1 >= 0) pB = p.values[coords.x + (coords.y - 1) * w.width];
    if (coords.y + 1 < p.height) pT = p.values[coords.x + (coords.y + 1) * w.width];

    Vector2D temp;
    temp.x = (pR - pL) * halfrdx;
    temp.y = (pT - pB) * halfrdx;

    *uNew = w.vectors[coords.x + coords.y * w.width];
    uNew->x = uNew->x - temp.x;
    uNew->y = uNew->y - temp.y; 
}

// BOUNDARY CONDITION
// no-slip (zero) velocity and pure Neumann pressure

void boundaryVectorField(Point2D coords, Point2D offset, Vector2D* bv, float scale, VectorField2D x) {
    (*bv).x = scale *  x.vectors[(coords.x + offset.x) + (coords.y + offset.y) * x.width].x;
    (*bv).y = scale *  x.vectors[(coords.x + offset.x) + (coords.y + offset.y) * x.width].y;
}

void boundaryScalarField(Point2D coords, Point2D offset, float* bv, float scale, ScalarField2D x) {
    (*bv) = scale *  x.values[(coords.x + offset.x) + (coords.y + offset.y) * x.width];
}


// GRAPHICS
GLFWwindow* initGraphics(int width, int height) {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return NULL;
    }

    GLFWwindow* window = glfwCreateWindow(width * SCALE, height * SCALE, "Fluid Simulation", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Faile to create GLFW window\n");
        glfwTerminate();
        return NULL;
    }

    glfwMakeContextCurrent(window);
    glViewport(0, 0, width * SCALE, height * SCALE);
    glOrtho(0, width * SCALE, 0, height * SCALE, -1, 1);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    return window;
}

void renderVectorField(VectorField2D field) {

    glBegin(GL_QUADS); // Begin rendering quads (squares)
    for (int y = 0; y < field.height; y++) {
        for (int x = 0; x < field.width; x++) {
            Vector2D v = field.vectors[x + y * field.width];
            
            float length = sqrtf(v.x * v.x + v.y * v.y);
            if (length > 0) {
                // Normalize the magnitude for coloring
                float normalizedLength = fminf(length / computeMaxMagnitude(field), 1.0f);
                
                // Use a color gradient (e.g., blue to red)
                glColor3f(normalizedLength, 0.0f, 1.0f - normalizedLength);

                // Compute the square's center
                float squareX = x * SCALE;
                float squareY = y * SCALE;

                // Render a square centered at (squareX, squareY)
                float squareSize = SCALE * 0.4f; // Size of the square as a fraction of the scale
                glVertex2f(squareX - squareSize, squareY - squareSize); // Bottom-left
                glVertex2f(squareX + squareSize, squareY - squareSize); // Bottom-right
                glVertex2f(squareX + squareSize, squareY + squareSize); // Top-right
                glVertex2f(squareX - squareSize, squareY + squareSize); // Top-left
            }
        }
    }
    glEnd();

}

void renderScalarField(ScalarField2D field) {
    
    glBegin(GL_POINTS);
    for (int y = 0; y < field.height; y++) {
        for (int x = 0; x < field.width; x++) {
            float value = field.values[x + y * field.width];
            float intensity = fminf(fmaxf(value, 0.0f), 1.0f);
            glColor3f(intensity, intensity, intensity);
            glVertex2f(x * SCALE, y * SCALE);
        }
    }
    glEnd();
}

// INITIALIZATION
VectorField2D createVectorField(int width, int length) {
    VectorField2D velocity;
    velocity.height = length;
    velocity.width = width;
    velocity.vectors = (Vector2D*)malloc(sizeof(Vector2D) * velocity.width * velocity.height);
    for (int i = 0; i < velocity.height; i++) {
        for (int j = 0; j < velocity.width; j++) {
            velocity.vectors[j + i * velocity.width].x = 0.0f;
            velocity.vectors[j + i * velocity.width].y = 0.0f;
        }
    }

    return velocity;
}

ScalarField2D createScalarField(int width, int length) {
    ScalarField2D field;
    field.height = length;
    field.width = width;
    field.values = (float*)malloc(sizeof(float) * field.width * field.height);
    for (int i = 0; i < field.height; i++) {
        for (int j = 0; j < field.width; j++) {
            field.values[j + i * field.width] = 0.0f;
        }
    }

    return field;
}

void initializeSmoothVelocityField(VectorField2D* field, int x_c, int y_c, float A, float sigma) {
    for (int y = 0; y < field->height; y++) {
        for (int x = 0; x < field->width; x++) {
            float dx = x - x_c;
            float dy = y - y_c;
            float distanceSquared = dx * dx + dy * dy;
            float magnitude = A * expf(-distanceSquared / (2 * sigma * sigma));
            field->vectors[x + y * field->width].x = magnitude;  // v_x
            field->vectors[x + y * field->width].y = magnitude;  // v_y
        }
    }
}



/*
void mainLoop(VectorField2D velocity, ScalarField2D pressure, float timestep) {
    // maybe some code to check the parameters in input

    // loop for \delta t -> timestep ?
        // loop for every point inside in the grid
            advectVectorField(point, sampleVectorFieldAtCoords(u, point), timestep, rdx, u, u); // ADVECTION
            // loop for jacobi iteration
                jacobiVectorField(point, sampleVectorFieldAtCoords(u, point), alpha, rBeta, u, u); // VISCOUS DIFFUSION
            forceApplication() // FORCE APPLICATION ?????
                               
            divergence(point, divergenceOfVelocity, halfrdx, u); // PROJECTION
            jacobiScalarField(point, sampleScalarFieldAtCoords(pressure, point), alfa, rBeta, pressure, divergenceOfVelocity); // PROJECTION
            gradient(point, sampleVectorFieldAtCoords(u, point), halfrdx, pressure, u); // PROJECTION

        // loop for every point outside in the grid
            // loop for every type of offset
                    bv = scale * sampleScalarFieldAtCoords(x, coords + offset);
boundaryVectorField(point, offset, sampleVectorFieldAtCoords(u, point), scale, u); // no-slip velocity
                boundaryScalarField(point, offset, sampleScalarFieldAtCoords(u, point), scale, p); // pure Neumann pressure


}
*/

// DEBUG
void printVectorField(VectorField2D field) {
    for (int y = 0; y < field.height; y++) {
        for (int x = 0; x < field.width; x++) {
            printf("[%f, %f] ", field.vectors[x + y * field.width].x, field.vectors[x + y * field.width].y);
        }
        printf("\n");
    }
}

void printScalarField(ScalarField2D field) {
    for (int y = 0; y < field.height; y++) {
        for (int x = 0; x < field.width; x++) {
            printf("[%5.2f] ", field.values[x + y * field.width]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {

    // Initialize parameters?

    // Initialize Graphics
    int width = W, height = W;
    GLFWwindow* window = initGraphics(width, height);
    if (!window)
        return 1;

    // Initializw Simulation Fields
    VectorField2D velocityOld = createVectorField(width, height);
    VectorField2D velocityNew = createVectorField(width, height);
    ScalarField2D pressureOld = createScalarField(width, height);
    ScalarField2D pressureNew = createScalarField(width, height);
    ScalarField2D div = createScalarField(width, height);

    // Set a small initial velocity at the center
    initializeSmoothVelocityField(&velocityOld, 3, 3, 1.0f, 1.0f);

    //float timestep = 0.016f; // 60 FPS
    float rdx = 1.0f / W;
    float halfrdx = 0.5f * rdx;
    float dx = 1 / rdx;
    float alpha, rBeta, time = 0;
    
    // Run Simulation Loop
    int z = 0;
    while(!glfwWindowShouldClose(window)) {
    //while (z++ < 15) {
        printf("\n\n---------- NEW ITERATION t: %f ----------\n\n", time);

        /*
        printf("INITIAL VELOCITY AND PRESSURE");
        printf("Velocity Old:\n");
        printVectorField(velocityOld);
        printf("\n\n");
        printf("Pressure Old:\n");
        printScalarField(pressureOld);
        printf("\n\n");
        printf("Velocity New:\n");
        printVectorField(velocityNew);
        printf("\n\n");
        printf("Pressure New:\n");
        printScalarField(pressureNew);
        */
        
        
        // ADVECTION
        for (int i = 1; i < height - 1; i++) {
            for (int j = 1; j < width - 1; j++) {
                Point2D coords;
                coords.x = j;
                coords.y = i;
                
                advectVectorField(coords, &velocityNew.vectors[j + i * width], TIMESTEP, rdx, velocityOld, velocityOld);
            }
        }
        memcpy(velocityOld.vectors, velocityNew.vectors, sizeof(Vector2D) * width * height);
        
        /*
        printf("AFTER ADVECTION - BEFORE DIFFUSION\n");
        //printf("Velocity Old:\n");
        //printVectorField(velocityOld);
        //printf("\n\n");
        //printf("Pressure Old:\n");
        //printScalarField(pressureOld);
        //printf("\n\n");
        printf("Velocity New:\n");
        printVectorField(velocityNew);
        printf("\n\n");
        //printf("Pressure New:\n");
        //printScalarField(pressureNew);
        */
        
        // DIFFUSION
        alpha = (- (dx * dx)) / (VISCOSITY * TIMESTEP);
        //alpha = (rdx * rdx) / (VISCOSITY * TIMESTEP);
        rBeta = 1.0f / (4.0f + alpha);
        for (int k = 0; k < NUM_JACOBI_ITERATIONS; k++) {
            for (int i = 1; i < height - 1; i++) {
                for (int j = 1; j < width - 1; j++) {
                    Point2D coords;
                    coords.x = j;
                    coords.y = i;

                    jacobiVectorField(coords, &velocityNew.vectors[j + i * width], alpha, rBeta, velocityOld, velocityOld);
                }
            }
            memcpy(velocityOld.vectors, velocityNew.vectors, sizeof(Vector2D) * width * height);;
        }
        /*
        printf("AFTER DIFFUSION - BEFORE FORCE APPLICATION\n");
        //printf("Velocity Old:\n");
        //printVectorField(velocityOld);
        //printf("\n\n");
        //printf("Pressure Old:\n");
        //printScalarField(pressureOld);
        //printf("\n\n");
        printf("Velocity New:\n");
        printVectorField(velocityNew);
        printf("\n\n");
        //printf("Pressure New:\n");
        //printScalarField(pressureNew);
        */

        // FORCE APPLICATION
        for (int y = 0; y < velocityNew.height; y++) {
            for (int x = 0; x < velocityNew.width; x++) {
                addExternalForce(&velocityNew.vectors[x + y * velocityNew.width], 0, 9.81f);
            }
        }
        //printf("AFTER FORCE APPLICATION - BEFORE PROJECTION\n");
        //printf("Velocity Old:\n");
        //printVectorField(velocityOld);
        //printf("\n\n");
        //printf("Pressure Old:\n");
        //printScalarField(pressureOld);
        //printf("\n\n");
        //printf("Velocity New:\n");
        //printVectorField(velocityNew);
        //printf("\n\n");
        //printf("Pressure New:\n");
        //printScalarField(pressureNew);

        
        // PROJECTION
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                Point2D coords;
                coords.x = j;
                coords.y = i;
                divergence(coords, &div.values[j + i * width], halfrdx, velocityOld);
            }
        }

        alpha = - (dx * dx);
        rBeta = 1.0f / 4;
        for (int k = 0; k < NUM_JACOBI_ITERATIONS; k++) {
            for (int i = 1; i < height - 1; i++) {
                for (int j = 1; j < width - 1; j++) {
                    Point2D coords;
                    coords.x = j;
                    coords.y = i;

                    jacobiScalarField(coords, &pressureNew.values[j + i * width], alpha, rBeta, pressureOld, div);
                }
            }
            memcpy(pressureOld.values, pressureNew.values, sizeof(float) * width * height);
        }

        for (int i = 1; i < height - 1; i++) {
            for (int j = 1; j < width - 1; j++) {
                Point2D coords;
                coords.x = j;
                coords.y = i;

                gradient(coords, &velocityNew.vectors[j + i * width], halfrdx, pressureNew, velocityOld);
            }
        }
        memcpy(velocityOld.vectors, velocityNew.vectors, sizeof(Vector2D) * width * height);
        
        printf("AFTER PROJECTION - BEFORE BOUNDARY CONDITIONS\n");
        //printf("Velocity Old:\n");
        //printVectorField(velocityOld);
        //printf("\n\n");
        //printf("Pressure Old:\n");
        //printScalarField(pressureOld);
        //printf("\n\n");
        printf("Velocity New:\n");
        printVectorField(velocityNew);
        printf("\n\n");
        printf("Pressure New:\n");
        printScalarField(pressureNew);

        // BOUNDARY CONDITIONS
        for (int i = 0; i < velocityNew.width; i++) {
            Point2D c1 = {i, 0};
            Point2D c1Offset = {0, 1};
            Point2D c2 = {i, velocityNew.height - 1};
            Point2D c2Offset = {0 , -1};

            boundaryVectorField(c1, c1Offset, &velocityNew.vectors[i], -1, velocityOld);
            boundaryVectorField(c2, c2Offset, &velocityNew.vectors[i + (velocityNew.height - 1) * velocityNew.width], -1, velocityOld);

            boundaryScalarField(c1, c1Offset, &pressureNew.values[i], -1, pressureOld);
            boundaryScalarField(c2, c2Offset, &pressureNew.values[i + (pressureNew.height - 1) * pressureNew.width], -1, pressureOld);
        }
        memcpy(velocityOld.vectors, velocityNew.vectors, sizeof(Vector2D) * width * height);
        memcpy(pressureOld.values, pressureNew.values, sizeof(float) * width * height);


        for (int i = 0; i < pressureNew.height; i++) {
            Point2D c1 = {0, i};
            Point2D c1Offset = {1, 0};
            Point2D c2 = {0, velocityNew.width - 1};
            Point2D c2Offset = {-1, 0};

            boundaryScalarField(c1, c1Offset, &pressureNew.values[i * pressureNew.width], 1, pressureOld);
            boundaryScalarField(c1, c1Offset, &pressureNew.values[(pressureNew.width - 1) + i * pressureNew.width], 1, pressureOld);

            boundaryVectorField(c1, c1Offset, &velocityNew.vectors[i * velocityNew.width], 1, velocityOld);
            boundaryVectorField(c1, c1Offset, &velocityNew.vectors[(velocityNew.width - 1) + i * velocityNew.width], 1, velocityOld);
        }
        memcpy(velocityOld.vectors, velocityNew.vectors, sizeof(Vector2D) * width * height);
        memcpy(pressureOld.values, pressureNew.values, sizeof(float) * width * height);

        printf("AFTER BOUNDARY CONDITIONS\n");
        //printf("Velocity Old:\n");
        //printVectorField(velocityOld);
        //printf("\n\n");
        //printf("Pressure Old:\n");
        //printScalarField(pressureOld);
        //printf("\n\n");
        printf("Velocity New:\n");
        printVectorField(velocityNew);
        printf("\n\n");
        printf("Pressure New:\n");
        printScalarField(pressureNew);

        // RENDER
        glClear(GL_COLOR_BUFFER_BIT);
        renderVectorField(velocityNew);
        //renderScalarField(pressureNew);
        glfwSwapBuffers(window);
        glfwPollEvents();

        time += TIMESTEP;
    }

    // Cleanup
    free(velocityOld.vectors);
    free(velocityNew.vectors);
    free(pressureOld.values);
    free(pressureNew.values);
    free(div.values);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

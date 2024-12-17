// OpenGL Graphics includes

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA FFT Libraries
//#include <cufft.h>

// CUDA helper functions
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <helper_cuda.h>

#include "defines.h"
#include "fluidsGL_kernels.h"

#define MAX_EPSILON_ERROR 1.0f

const char *sSDKname = "fluidsGL";

void cleanup(void);
void reshap(int x, int y);

// CUFFT plan handle ????

typedef struct {
    float x;
    float y;
} cData;

cData *hvfield = NULL;
cData *dvfield = NULL;
static int wWidth = 512;
static int wHeight = 512;

static int clicked = 0;
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

// Particle data
GLuint vbo = 0; // OpenGL vertex buffer object
static cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange
static cData *particles = NULL; // particle positions in host memory
static int lastx = 0, lasty = 0;

// Texture pitch
size_t tPitch = 0; 

char *ref_file = NULL;
bool g_bQAAddTestForce = true;
int g_iFrameToCompare = 100;
int g_TotalErrors = 0;

bool g_bExitEscc = false;

// CheckFBO/BackBuffer class object
CheckRender *g_CheckRender = NULL;

void autoTest(char **);

/*
 *  FUNCTIONS FOR FLUID SIMULATION
 */

void simulateFluids(void) {
    /*
     *  advectVelocity()
     *  diffuese()
     *  project()
     *  updateVelocity()
     *  advectParticles() ???????
     */
}

void display(void) {
    if (!ref_file) {
        sdkStartTimer(&timer);
        simulateFluids();
    }

    // render points from vertex buffer
}
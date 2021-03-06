// belongs to 2DFDTD CUDA V2.02 - VBO

#include "global.h"
#include "GL/glew.h"
#include <GL/GL.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "grid_2d.h"
#include "cuda_functions.h"

// rendering callbacks, functions in graphics.cpp
void idle();
void keyboard(unsigned char key, int x, int y);
void reshape(int w, int h);
void setImageAndWindowSize();			// Self-explanatory.  Why am I writing this.
bool deallocateArrays();				// used to free() all host arrays
bool runFdtdWithFieldDisplay(int argc, char** argv);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void initGLVbo(int argc, char **argv);
void runIterationsAndDisplayVbo();
bool runFdtdWithFieldDisplayVbo(int argc, char** argv);
void CleanupVbo(int iExitCode);
void renderCudaVbo(int drawMode);	// This is used instead of displayTextureImage (as in 2DFDTD CUDA V2.01)
void initCudaVbo();
void cleanupCudaVbo();
void deleteVBO(mappedBuffer_t* mbuf);
void createVBO(mappedBuffer_t* mbuf);



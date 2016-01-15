// This file is intended for external variable declarations so that OpenGL functions 
// have access to program variables without directly passing them.

// Belongs to CARTESIAN 2DFDTD CUDA V2.02

#pragma once
#include <cstdlib>
#include <cstdio>
#include <helper_math.h>
#ifndef _global_h
#define _global_h
#define USE_CUDA3

#include "grid_2d.h"
#include <GL/glew.h>
#define PI 3.14159265359

typedef struct {									// This structure contains vbo data
	GLuint vbo;
	GLuint typeSize;
#ifdef USE_CUDA3
	struct cudaGraphicsResource *cudaResource;
#else
	void* space;
#endif
} mappedBuffer_t;

//__constant__ unsigned int  dvrgb[256];	
extern const int TILE_SIZE;					// Tile size, relates closely to size of a block.  

// vbo variables
extern mappedBuffer_t vertexVBO;
extern mappedBuffer_t colorVBO;
extern const unsigned int RestartIndex;
extern float4 *dptr;					// The vertex part of the vbo - generated only once
extern uchar4 *cptr;					// The color part of the vbo - generated each time loop

extern Grid *g;
extern double *dev_hx;					// Now the global device pointer for field Hx
extern double *dev_chxh;				// Global device pointer for Chxh
extern double *dev_chxe;				// Same
extern double *dev_hy;					// Now the global device pointer for field Hy
extern double *dev_chyh;				// Same
extern double *dev_chye;				// Same
extern double *dev_ez;					// Now the global device pointer for field Ez
extern double *dev_cezh;				// Same
extern double *dev_ceze;				// Same
extern float *dev_ez_float;				// Copy of dev_ez but in single precision
extern float *dev_hx_float;				// Copy of dev_hx but in single precision
extern float *dev_hy_float;				// Copy of dev_hy but in single precision
extern double *dev_ezleft;				// Left side of the Ez grid, for the 2nd Order Mur ABC
extern double *dev_ezright;				// Right side
extern double *dev_eztop;				// Top side
extern double *dev_ezbottom;			// Bottom side

// Note for all the externs declared below:  they have no location in memory until defined somewhere else (or here).  
// Extern <variable type> just declares the variable globally to the program, but it does not exist until
// it has been defined.
extern const double e0;					// electric permittivity of free space
extern const double er;                 // Relative electric permittivity
extern const double u0;					// magnetic permeability of free space
extern const double ur;                 // relative magnetic permeability
extern const double imp0;				// impedance of free space
extern const double mag_cond;           // Magnetic conductivity
extern const double el_cond;            // Electric conductivity
extern const double c;

extern float global_min_field;			// calculated by find_min_max_on_gpu
extern float global_max_field;			// calculated by find_min_max_on_gpu
extern float *dvminimum_field_value;	// Both of these are passed to the find-min-max-gpu functions
extern float *dvmaximum_field_value;	// to get proper min/max field values for color-scaling
extern bool show_Ez;					// Used as a flag in visualization
extern int plotting_step;				// Used in IterationAndDisplay; every plotting_step steps arrays will be displayed via OpenGL

extern int iGLUTWindowHandle;			// handle to the GLUT window
#endif
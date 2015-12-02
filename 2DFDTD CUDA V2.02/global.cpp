// This file belongs to 2DFDTD CUDA V2.02

#include "global.h"
float *dev_ez_float;
float *dev_hx_float;
float *dev_hy_float;
double *dev_ez;
double *dev_hx;
double *dev_hy;
double *dev_chxh;
double *dev_chxe;
double *dev_chyh;
double *dev_chye;
double *dev_cezh;
double *dev_ceze;
float *dvminimum_field_value;
float *dvmaximum_field_value;
Grid *g = new Grid;
int plotting_step;

//  Vertex buffer stuff
float4 *dptr;					// The vertex part of the vbo - generated only once
uchar4 *cptr;					// The color part of the vbo - generated each time loop
uint *iptr;						// Not sure what this is yet.

const double c = 299792458.0;
double src_f = 11e3;                 // Frequency of the source (for a sine or cosine)
double lambda = c / src_f;           // Wavelength of the source (for a sine or cosine)
double dx = 1.0;
double Sc = 1 / sqrt(2.0);
double dt = Sc*dx / c;
double N_lambda = lambda / dx;
//double Sc = c*dt / dx;


const unsigned int RestartIndex = 0xffffffff;		// Used for primitive restart (VBO)
mappedBuffer_t vertexVBO = { NULL, sizeof(float4), NULL };
mappedBuffer_t colorVBO = { NULL, sizeof(uchar4), NULL };

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
double *dev_ezright;
double *dev_ezleft;
double *dev_eztop;
double *dev_ezbottom;
float *dvminimum_field_value;
float *dvmaximum_field_value;
Grid *g = new Grid;
int plotting_step;

//  Vertex buffer stuff
float4 *dptr;					// The vertex part of the vbo - generated only once
uchar4 *cptr;					// The color part of the vbo - generated each time loop
//uint *iptr;						// Not sure what this is yet.

const int TILE_SIZE = 32;			 // Tile size, relates closely to size of a block.  
const double c = 299792458.0;
const double e0 = 8.85418782e-12;		// electric permittivity of free space
const double er = 1;                    // Relative electric permittivity
const double u0 = 4 * PI *1e-7;			// magnetic permeability of free space
const double ur = 1.0;                  // relative magnetic permeability
const double imp0 = sqrt(u0 / e0);		// impedance of free space
const double mag_cond = 0.0;            // Magnetic conductivity
const double el_cond = 0.0;             // Electric conductivity


const unsigned int RestartIndex = 0xffffffff;		// Used for primitive restart (VBO)
mappedBuffer_t vertexVBO = { NULL, sizeof(float4), NULL };
mappedBuffer_t colorVBO = { NULL, sizeof(uchar4), NULL };

#ifndef _parameters_h
#define _parameters_h

#include <math.h>
#define PI 3.14159265359
const int maxTime = 1500;				// number of time steps         
const int M = 1024;						// steps in x-direction
const int N = 1024;						// steps in y-direction
const int TILE_SIZE = 32;				// Tile size, relates closely to size of a block.  
//const double c = 299792458.0;			// speed of light in vacuum					
const double e0 = 8.85418782e-12;		// electric permittivity of free space
const double er = 1;                  // Relative electric permittivity
const double u0 = 4 * PI *1e-7;			// magnetic permeability of free space
const double ur = 1.0;                  // relative magnetic permeability
const double imp0 = sqrt(u0 / e0);		// impedance of free space
const double mag_cond = 0.0;            // Magnetic conductivity
const double el_cond = 0.0;             // Electric conductivity
//const double src_f = 11e7;                 // Frequency of the source (for a sine or cosine)
//double N_lambda = 25;                // Points per wavelength Nl = lambda * dx
//const double lambda = c / src_f;           // Wavelength of the source (for a sine or cosine)
//const double Sc = 1 / ((double)sqrt(2.0));
// The following variables are for material geometry and should probably
// be placed in a separate file related to the source  
const int barpos_x1 = 190;				// 2 * M / 5;
const int barpos_x2 = 290;				//3 * M / 5;
const int barpos_y1 = 400;				// 2 * N / 3 - N / 40;
const int barpos_y2 = 450;				// 2 * N / 3 + N / 40;
const int src_pos_x = (int)(0.85 * M);
const int src_pos_y = (int)(N / 2);
const int src_pos_x2 = (int)(0.35 * M);
const int src_pos_y2 = (int)(4*N/6);
const int r1 = M / 4;					// radius of inner PEC
const int r2 = M / 2;					// radius of outer PEC

// Other variables

#endif
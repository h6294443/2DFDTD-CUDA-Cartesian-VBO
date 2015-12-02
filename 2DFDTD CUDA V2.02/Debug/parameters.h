#define PI 3.14159265359
const int maxTime = 100;		    // number of time steps         
const int M = 20;					// steps in x-direction
const int N = 20;					// steps in y-direction
const int r1 = M / 4;				// radius of inner PEC
const int r2 = M / 2;				// radius of outer PEC
const double c = 299792458.0;		// speed of light in vacuum					
const double e0 = 8.85418782e-12;	// electric permittivity of free space
const double u0 = 4 * PI *1e-7;     // magnetic permeability of free space
const double imp0 = sqrt(u0 / e0);  // impedance of free space
const double Sc = 1 / ((float) sqrt(2.0));
const int N_lambda = 25;
const int barpos_x1 = 190; // 2 * M / 5;
const int barpos_x2 = 290; //3 * M / 5;
const int barpos_y1 = 400; // 2 * N / 3 - N / 40;
const int barpos_y2 = 450; // 2 * N / 3 + N / 40;
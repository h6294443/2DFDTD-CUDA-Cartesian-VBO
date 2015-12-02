/*	Function to implement a Ricker wavelet and cosine source.  
    This is a traveling wave version of the function so ezInc() takes arguments of both time and space	*/
/*	This source file belongs to project "Ch8_8.4 (TMz Example)"	*/
#include "global.h"
#include "source.h"
#include "parameters.h"
//#include "grid-2d.h"

#define _USE_MATH_DEFINES
#include <math.h>

static double cdtds, ppw = 0;

//	Initialize source-function variables	
void ezIncInit(Grid *g) {

	//printf("Enter the points per wavelength for source: ");
	//scanf_s(" %lf", &ppw);
	ppw = 250;
	cdtds = Sc;
	return;
}

//	Calculate source function at given time and location	
double ezIncRicker(int time, double location) {
	double arg;

	if (ppw <= 0) {
		fprintf(stderr,
			"ezInc: ezIncInit() must be called before ezInc.\n"
			"		Points per wavelength must be positive.\n");
		exit(-1);
	}

	arg = M_PI * ((cdtds * time - location) / ppw - 1.0);
	arg = arg * arg;

	return (1.0 - 2.0 * arg) * exp(-arg);
}

double ezIncCos(double time) {
	double arg;

	if (ppw <= 0) {
		fprintf(stderr, "ezInc: ezIncInit() must be called before ezInc.\n"
			"Points per wavelength must be positive.\n");
		exit(-1);
	}

	arg = cos(2 * M_PI * 1 * time / ppw);
	return arg;
}





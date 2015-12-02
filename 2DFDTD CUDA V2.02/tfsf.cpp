/*	This source file belongs to project "Ch8_8.6 (TMz TFSF Boundary Example)"	*/
/*	This file implements a TFSF boundary for a TMz grid.  The incident field is */
/*	assumed to propagate alond the x-direction and is calculated using an		*/
/*	auxiliary 1D simulation														*/

#include <string.h>
#include "macros.h"
#include "grid_2d.h"
#include "source.h"
#include "cuda_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include "parameters.h"

static int firstX = 0, firstY,				// Indices for first point in TF region
lastX, lastY;								// Indices for last point in TF region

static Grid *g1 = new Grid;							// This is the 1D auxiliary grid

void tfsfInit(Grid *g)
{
	//Grid *g1 = new Grid;					// Allocate memory for 1D aux grid
	gridInit(g1);
	memcpy(g1, g, sizeof(Grid));			// Copy all information from 2D grid
	g1->type = oneDGrid;					// Initialize the 1D aux grid

	printf("Grid is %d by %d cells.\n", SizeX, SizeY);
	printf("Enter indices for first point in TF region: ");
	scanf(" %d %d", &firstX, &firstY);
	printf("Enter indices for last point in TF region: ");
	scanf(" %d %d", &lastX, &lastY);

	ezIncInit(g1);							// Initialize source function

	return;
}

void tfsfUpdate(Grid *g)
{
	int mm, nn;

	/*	First check if tfsfInit() has been called first	*/
	if (firstX <= 0)
	{
		fprintf(stderr,
			"tfsfUpdate: tfsfInit must be called before tfsfUpdate.\n"
			"			 Boundary location must be set to positive value.\n");
		exit(-1);
	}

	/*	Correct Hy along left edge	*/
	mm = firstX - 1;
	for (nn = firstY; nn <= lastY; nn++)
		Hy(mm, nn) -= Chye(mm, nn) * Ez1G(g1, mm + 1);

	/*	Correct Hy along right edge	*/
	mm = lastX;
	for (nn = firstY; nn <= lastY; nn++)
		Hy(mm, nn) += Chye(mm, nn) * Ez1G(g1, mm);

	/*	Correct Hx on bottom edge	*/
	nn = firstY - 1;
	for (mm = firstX; mm <= lastX; mm++)
		Hx(mm, nn) += Chxe(mm, nn) * Ez1G(g1, mm);

	/*	Correct Hx along top edge	*/
	nn = lastY;
	for (mm = firstX; mm <= lastX; mm++)
		Hx(mm, nn) -= Chxe(mm, nn) * Ez1G(g1, mm);

	/*cudaError_t cudaStatus = updateH2D_CUDA(g1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "updateH2D_CUDA failed during tfsfUpdate.");
		
	}
	cudaStatus = updateE2D_CUDA(g1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "updateE2D_CUDA failed during tfsfUpdate.");
		
	}*/
	
	Ez1G(g1, 0) = ezIncRicker(TimeG(g1), 0.0);	// Set source node
	TimeG(g1)++;								// Increment time in the 1D aux grid

	/*	Correct Ez adjacent to TFSF boundary	*/
	/*	Correct Ez field along the left edge	*/
	mm = firstX;
	for (nn = firstY; nn <= lastY; nn++)
		Ez(mm, nn) -= Cezh(mm, nn) * Hy1G(g1, mm - 1);

	/*	Correct Ez field along the right edge	*/
	mm = lastX;
	for (nn = firstY; nn <= lastY; nn++)
		Ez(mm, nn) += Cezh(mm, nn) * Hy1G(g1, mm);

	/*	No need to correct Ez along top and		*/
	/*	bottom since incident Hx is zero.		*/

	return;
}



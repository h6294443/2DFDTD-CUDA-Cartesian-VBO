/*	This source file belongs to 2DFDTD CUDA Cartesian VBO	*/
/*	This source file implements a function to apply a second-order ABC to a TMz grid	*/
/*  January 9, 2016
*/
#include "global.h"
#include "cuda_functions.h"
#include "source.h"
#include <math.h>
#include "grid_2d.h"


/*	Second-order ABC for TMz grid	*/

/*__________________________________________________________________________*/
/*	Define macros for arrays that store the previous values of the fields.	*/
/*	For each one of these arrays the three arguments are as follows:		*/
/*																			*/
/*	first argument:		spatial displacement from the boundary				*/
/*	second argument:	displacement back in time							*/
/*	third argument:		distance from either bottom (if EzLeft or EzRight)	*/
/*						or left (if EzTop or EzBottom) side of grid			*/
/*__________________________________________________________________________*/

#define EzLeft(M, Q, N)		g->ezLeft[(N) * 6 + (Q) * 3 + (M)]
#define EzRight(M, Q, N)	g->ezRight[(N) * 6 + (Q) * 3 + (M)]
#define EzTop(N, Q, M)		g->ezTop[(M) * 6 + (Q) * 3 + (N)]
#define EzBottom(N, Q, M)	g->ezBottom[(M) * 6 + (Q) * 3 + (N)]
#define Ez(M,N)				g->Ez[(M) + (N) * g->M]
#define SizeX				g->M
#define SizeY				g->N
#define dEzLeft(M,Q,N)		dEzLeft[(N) * 6 + (Q) * 3 + (M)]
#define dEzRight(M,Q,N)		dEzRight[(N) * 6 + (Q) * 3 + (M)]
#define dEzTop(N,Q,M)		dEzTop[(M) * 6 + (Q) * 3 + (N)]
#define dEzBottom(N,Q,M)	dEzBottom[(M) * 6 + (Q) * 3 + (N)]
#define dEz(M,N)			dEz[(M)+(N)*width]

static int initDone = 0;
static double coef0, coef1, coef2;


__global__ void Mur_ABC_kernel(double *dEz, double *dEzLeft, double *dEzRight, double *dEzBottom, double *dEzTop, double coef0, double coef1, double coef2, int width, int height)
{
	// Map from threadIdx/blockIdx to cell position
	int offset = blockIdx.x * blockDim.x + threadIdx.x;	// Calculate linear offset for 1-D unrolled array
	int j = offset / width;							// Creates a virtual row index for the 1-D case, needed for odd/even row check
	int i = offset % width;							// Keeping both offset and i for clarity

	if (offset < width*height) {
		
		// left side first
		if (i == 0) {
			// calculate new  outside Ez
			dEz[offset] = coef0 * (dEz[offset + 2] + dEzLeft(0, 1, j)) + coef1 * (dEzLeft(0, 0, j) + dEzLeft(2, 0, j)
				 - dEz[offset + 1] - dEzLeft(1, 1, j)) + coef2 * dEzLeft(1, 0, j) - dEzLeft(2, 1, j);

			// memorize old fields now
			for (int mm = 0; mm < 3; mm++) {
				dEzLeft(mm, 1, j) = dEzLeft(mm, 0, j);
				dEzLeft(mm, 0, j) = dEz[offset + mm];
			}
		}
		// right side
		if (i == width - 1) {
			// the right side requires a -1 offset for some reason.  I don't know why as of 1/10/2016
			dEz[offset-1] = coef0 * (dEz[offset - 3] + dEzRight(0, 1, j)) + coef1 * (dEzRight(0, 0, j) + dEzRight(2, 0, j)
					- dEz[offset-2] - dEzRight(1, 1, j)) + coef2 * dEzRight(1, 0, j) - dEzRight(2, 1, j);

			// memorize old fields
			for (int mm = 0; mm < 3; mm++){
				dEzRight(mm, 1, j) = dEzRight(mm, 0, j);
				dEzRight(mm, 0, j) = dEz[offset - mm-1]; // offset of -1 present here, too 				
			}
		}		

		// bottom side
		if (j == 0) {
			// calculate new outside Ez
			dEz[offset] = coef0 * (dEz[offset+2*width] + dEzBottom(0, 1, i)) + coef1 * (dEzBottom(0, 0, i) + dEzBottom(2, 0, i)
				 - dEz[offset+width] - dEzBottom(1, 1, i)) + coef2 * dEzBottom(1, 0, i) - dEzBottom(2, 1, i);

			//	memorize old fields
			for (int nn = 0; nn < 3; nn++) {
				dEzBottom(nn, 1, i) = dEzBottom(nn, 0, i);
				dEzBottom(nn, 0, i) = dEz[offset + nn * width];
			}
		}
		// top side
		if (j == height - 1) {
			// calculate new outside Ez
			dEz[offset] = coef0 * (dEz[offset - 2*width] + dEzTop(0, 1, i)) + coef1 * (dEzTop(0, 0, i) + dEzTop(2, 0, i) 
				- dEz[offset-width] - dEzTop(1, 1, i)) + coef2 * dEzTop(1, 0, i) - dEzTop(2, 1, i);

			//	Now memorize old fields
			for (int nn = 0; nn < 3; nn++) {
				dEzTop(nn, 1, i) = dEzTop(nn, 0, i);
				dEzTop(nn, 0, i) = dEz[offset - nn * width];
			}
		}		
	}


}

void abcInit(Grid *g)
{
	double temp1, temp2;
	initDone = 1;
		
	for (int i = 0; i < (g->M * 6); i++) {
		g->eztop[i] = 0.0;
		g->ezbottom[i] = 0.0;
	}
	for (int i = 0; i < (g->N * 6); i++) {
		g->ezleft[i] = 0.0;
		g->ezright[i] = 0.0;
	}
		
	//	Calculate ABC coefficients		
	temp1 = sqrt(g->cezh[0] * g->chye[0]);
	temp2 = 1.0 / temp1 + 2.0 + temp1;
	coef0 = -(1.0 / temp1 - 2.0 + temp1) / temp2;
	coef1 = -2.0 * (temp1 - 1.0 / temp1) / temp2;
	coef2 = 4.0 * (temp1 + 1.0 / temp1) / temp2;

	return;
}

void abc(Grid *g)
{
	// Calculate CUDA grid dimensions.  Block dimension fixed at 32x32 threads
	int TILE_SQUARED = TILE_SIZE * TILE_SIZE;
	int Bx = (TILE_SQUARED - 1 + g->nCells) / TILE_SQUARED;
	dim3 BLK(Bx, 1, 1);									// Grid-block dimension for the 1-D case
	dim3 THD(TILE_SQUARED, 1, 1);							// Thread-block dimension for the 1-D case

	//HxHyUpdate_Kernel << <BLK, THD >> >(dev_hx, dev_chxh, dev_chxe, dev_hy, dev_chyh, dev_chye, dev_ez, g->M, g->N);
	Mur_ABC_kernel<<<BLK,THD>>>(dev_ez, dev_ezleft, dev_ezright, dev_ezbottom, dev_eztop, coef0, coef1, coef2, g->M, g->N);
	checkErrorAfterKernelLaunch();						// Check for any errors launching the kernel
	deviceSyncAfterKernelLaunch();						// Do a device sync 

	//	ABC at left side of grid	
	/*for (nn = 0; nn < g->sizeY; nn++)
	{
		Ez(0, nn) = coef0 * (Ez(2, nn) + EzLeft(0, 1, nn)) + coef1 * (EzLeft(0, 0, nn) + EzLeft(2, 0, nn) - Ez(1, nn) - EzLeft(1, 1, nn))
			+ coef2 * (EzLeft(1, 0, nn) - EzLeft(2, 1, nn));

		//	Now memorize old fields
		for (mm = 0; mm < 3; mm++)
		{
			EzLeft(mm, 1, nn) = EzLeft(mm, 0, nn);
			EzLeft(mm, 0, nn) = Ez(mm, nn);
		}
	}

	//	ABC at right side of grid	
	for (nn = 0; nn < SizeY; nn++)
	{
		Ez(SizeX - 1, nn) = coef0 * (Ez(SizeX - 3, nn) + EzRight(0, 1, nn)) + coef1 * (EzRight(0, 0, nn) + EzRight(2, 0, nn) - Ez(SizeX - 2, nn) - EzRight(1, 1, nn))
			+ coef2 * EzRight(1, 0, nn) - EzRight(2, 1, nn);

		//	Now memorize old fields
		for (mm = 0; mm < 3; mm++)
		{
			EzRight(mm, 1, nn) = EzRight(mm, 0, nn);
			EzRight(mm, 0, nn) = Ez(SizeX - 1 - mm, nn);
		}
	}

	//	ABC at bottom of the grid	
	for (mm = 0; mm < g->sizeX; mm++)
	{
		Ez(mm, 0) = coef0 * (Ez(mm, 2) + EzBottom(0, 1, mm)) + coef1 * (EzBottom(0, 0, mm) + EzBottom(2, 0, mm) - Ez(mm, 1) - EzBottom(1, 1, mm))
			+ coef2 * EzBottom(1, 0, mm) - EzBottom(2, 1, mm);

		//	Now memorize old fields
		for (nn = 0; nn < 3; nn++)
		{
			EzBottom(nn, 1, mm) = EzBottom(nn, 0, mm);
			EzBottom(nn, 0, mm) = Ez(mm, nn);
		}
	}


	//	ABC at the top of the grid	
	for (mm = 0; mm < g->sizeX; mm++)
	{
		Ez(mm, SizeY - 1) = coef0 * (Ez(mm, SizeY - 3) + EzTop(0, 1, mm)) + coef1 * (EzTop(0, 0, mm) + EzTop(2, 0, mm) - Ez(mm, SizeY - 2) - EzTop(1, 1, mm))
			+ coef2 * EzTop(1, 0, mm) - EzTop(2, 1, mm);

		//	Now memorize old fields
		for (nn = 0; nn < 3; nn++)
		{
			EzTop(nn, 1, mm) = EzTop(nn, 0, mm);
			EzTop(nn, 0, mm) = Ez(mm, SizeY - 1 - nn);
		}
	}*/

	return;
}


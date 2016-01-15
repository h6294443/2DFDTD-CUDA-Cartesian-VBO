#include "grid_2d.h"
#include "global.h"
#include <stdio.h>

void gridInit(Grid *g) {
	
	// Step 1: Specify source frequency	and type
	printf("Enter source frequency in Hz (Engineering/Scientific notation ok): ");
	scanf_s(" %lf", &g->src_f);
	printf("\nEnter source type: \n");
	printf("0 = sine\n");
	printf("1 = cosine\n");
	printf("2 = Ricker wavelet\n");
	scanf_s(" %i", &g->src_type);
	if (g->src_type < 0) g->src_type = 0;
	if (g->src_type > 2) g->src_type = 2;

	// Step 2: Calculate wavelength
	g->lambda = c / g->src_f;           // Wavelength of the source (for a sine or cosine)

	// Step 3: Specify Sc:
	printf("\n\nEnter desired Courant number (0.1-0.7): ");
	scanf_s(" %lf", &g->cdtds);
	if (g->cdtds < 0.1) g->cdtds = 0.1;
	if (g->cdtds > 0.7) g->cdtds = 0.8;

	// Step 4: Specify physical domain size in meters
	printf("\n\nEnter the domain width (X) in meters: ");
	scanf_s(" %lf", &g->DOMX);
	printf("\n\nEnter the domain height (Y) in meters: ");
	scanf_s(" %lf", &g->DOMY);

	// Step 5: Specify desired points-per-wavelength N_lambda and simulation time
	printf("\n\nEnter points-per-wavelength (can be a float): ");
	scanf_s(" %lf", &g->N_lambda);
	printf("\n\nEnter simulation time steps: ");
	scanf_s(" %i", &g->maxTime);

	// Step 6: Calculate dx (this may not be dx as defined)
	g->dx = g->lambda / g->N_lambda;				// This is the largest distance possible within one hexagon - from one point to the opposing pointy point

	// Step 7: Calculate dt
	g->dt = g->cdtds*g->dx / c;

	// Step 8: Calculate M and N
	g->M = (int)(g->DOMX / g->dx);
	g->N = (int)(g->DOMY / g->dx);
	g->nCells = g->M*g->N;

	// Step 8: Specify source position (and soon, type)
	//const int src_pos_x = (int)(0.15*M);
	//const int src_pos_y = (int)(0.5*N);
	g->src_x = 0.75 * g->DOMX;
	g->src_y = 0.75 * g->DOMY;
	g->src_i = (int)(g->src_x / g->dx);	// source position in the Ez array
	g->src_j = (int)(g->src_y / g->dx);

	// Step 9: Specify desired slowdown, if any.
	//const int slowdown = 25;
	
	g->time = 0;
	
	g->type = tmZGrid;
	
	int m, n;
	
	g->hx = new double[g->M * (g->N - 1)];
	g->chxh = new double[g->M * (g->N - 1)];
	g->chxe = new double[g->M * (g->N - 1)];
	g->hy = new double[(g->M - 1) * g->N];
	g->chyh = new double[(g->M - 1) * g->N];
	g->chye = new double[(g->M - 1) * g->N];
	g->ez = new double[g->M * g->N];
	g->ceze = new double[g->M * g->N];
	g->cezh = new double[g->M * g->N];
	g->ez_float = new float[g->M * g->N];

	// The following four arrays are for storing domain boundary
	// values for the ABC (2nd Order Mur)
	g->ezright = new double[g->N * 6];
	g->ezleft = new double[(g->N * 6)];
	g->eztop = new double[(g->M * 6)];
	g->ezbottom = new double[(g->M * 6)];

	printf("\n\nSimulation parameters\n");
	printf("Domain size: %g m by %g m and %i by %i cells (M by N)\n", g->DOMX, g->DOMY, g->M, g->N);
	printf("Source frequency = %3g Hz, wavelength = %g m, and ppw = %g\n", g->src_f, g->lambda, g->N_lambda);
	printf("dx = %g m, dt = %g s, Sc = %g \n", g->dx, g->dt, g->cdtds);
	printf("Source is at (%3g, %3g) meters.\n\n", g->src_x, g->src_y);
	
	double chxh = (1 - (mag_cond*g->dt) / (2 * u0*ur)) / (1 + (mag_cond*g->dt) / (2 * u0*ur));
	double chxe = 1 / (1 + (mag_cond*g->dt) / (2 * u0*ur)) * g->dt / (u0*ur*g->dx);
	double chyh = (1 - (mag_cond*g->dt) / (2 * u0*ur)) / (1 + (mag_cond*g->dt) / (2 * u0*ur));
	double chye = 1 / (1 + (mag_cond*g->dt) / (2 * u0*ur)) * g->dt / (u0*ur*g->dx);
	double ceze = (1 - (el_cond*g->dt) / (2 * e0*er)) / (1 + (el_cond*g->dt) / (2 * e0*er));
	double cezh = 1 / (1 + (el_cond*g->dt) / (2 * e0*er)) * g->dt / (e0*er*g->dx);

	//	Set the magnetic-field Hx update coefficients and initialize Hx
	for (m = 0; m < g->N - 1; m++) {
		for (n = 0; n < g->M; n++) {
			int offset = n + m * g->M;
			g->chxh[offset] = chxh;
			g->chxe[offset] = chxe;
			g->hx[offset] = 0.0;
		}
	}

	//	Set the magnetic-field Hy update coefficients and initialize Hy
	for (m = 0; m < g->N-1; m++) {
		for (n = 0; n < g->M; n++) {
			int offset = n + m * g->M;
			g->chyh[offset] = chyh;
			g->chye[offset] = chye;
			g->hy[offset] = 0.0;			
		}
	}

	// Set electric field and coefficients
	for (m = 0; m < g->N; m++) {
		for (n = 0; n < g->M; n++) {
			int offset = n + m * g->M;
			g->ceze[offset] = ceze;
			g->cezh[offset] = cezh;
			g->ez[offset] = 0;
			g->ez_float[offset] = 0;
		}
	}

	double xc = g->DOMX / 2;
	double yc = g->DOMY / 2;
	double r1 = 0.175 * g->DOMX;
	double r2 = 0.9 * g->DOMX;

	// Set up circular PEC 
	for (m = 0; m < g->N; m++) {
		for (n = 0; n < g->M; n++) {
			int offset = n + m * g->M;
			double x_current = n * g->dx;
			double y_current = m * g->dx;
			double dx = x_current - xc;
			double dy = y_current - yc;
			double check = sqrt(dx*dx + dy*dy);
			if (check < r1 ) {//|| check > r2){
				g->ceze[offset] = 0.0;
				g->cezh[offset] = 0.0;
			}
		}
	}
}
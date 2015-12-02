#include "parameters.h"
#include "grid_2d.h"
#include "global.h"
#include <stdio.h>

void gridInit(Grid *g) {
	g->sizeX = M;
	g->sizeY = N;
	g->time = 0;
	g->maxTime = maxTime;
	g->type = tmZGrid;
	g->cdtds = Sc;      // fixed value in parameter.h
	int m, n;
	
	g->hx	= new double[M * (N - 1)];
	g->chxh = new double[M * (N - 1)];
	g->chxe = new double[M * (N - 1)];
	g->hy	= new double[(M - 1) * N];
	g->chyh = new double[(M - 1) * N];
	g->chye = new double[(M - 1) * N];
	g->ez	= new double[M    *    N];
	g->ceze = new double[M    *    N];
	g->cezh = new double[M    *    N];
	g->ez_float = new float[M * N];

	printf("Lambda: %g m\n", lambda);
	printf("N-lambda: %g\n", N_lambda);
	printf("dx: %g m\n", dx);
	printf("dt: %g s\n", dt);
	printf("Sc: %g\n", Sc);
	
	double chxh = (1 - (mag_cond*dt) / (2 * u0*ur)) / (1 + (mag_cond*dt) / (2 * u0*ur));
	double chxe = 1 / (1 + (mag_cond*dt) / (2 * u0*ur)) * dt / (u0*ur*dx);
	double chyh = (1 - (mag_cond*dt) / (2 * u0*ur)) / (1 + (mag_cond*dt) / (2 * u0*ur));
	double chye = 1 / (1 + (mag_cond*dt) / (2 * u0*ur)) * dt / (u0*ur*dx);
	double ceze = (1 - (el_cond*dt) / (2 * e0*er)) / (1 + (el_cond*dt) / (2 * e0*er));
	double cezh = 1 / (1 + (el_cond*dt) / (2 * e0*er)) * dt / (e0*er*dx);

	//	Set the magnetic-field Hx update coefficients and initialize Hx
	for (m = 0; m < g->sizeY - 1; m++) {
		for (n = 0; n < g->sizeX; n++) {
			int offset = n + m * g->sizeX;
			g->chxh[offset] = chxh;
			g->chxe[offset] = chxe;
			g->hx[offset] = 0.0;
		}
	}

	//	Set the magnetic-field Hy update coefficients and initialize Hy
	for (m = 0; m < g->sizeY-1; m++) {
		for (n = 0; n < g->sizeX; n++) {
			int offset = n + m * g->sizeX;
			g->chyh[offset] = chyh;
			g->chye[offset] = chye;
			g->hy[offset] = 0.0;			
		}
	}

	// Set electric field and coefficients
	for (m = 0; m < g->sizeY; m++) {
		for (n = 0; n < g->sizeX; n++) {
			int offset = n + m * g->sizeX;
			g->ceze[offset] = ceze;
			g->cezh[offset] = cezh;
			g->ez[offset] = 0;
			g->ez_float[offset] = 0;
		}
	}
}
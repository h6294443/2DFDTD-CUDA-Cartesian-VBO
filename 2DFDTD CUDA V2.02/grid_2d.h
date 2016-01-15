#ifndef _GRID_2D_H
#define _GRID_2D_H

enum GRIDTYPE { oneDGrid, teZGrid, tmZGrid, threeDGrid };

struct Grid {
	double *hx, *chxh, *chxe;
	double *hy, *chyh, *chye;
	double *hz, *chzh, *chze;
	double *ex, *cexe, *cexh;
	double *ey, *ceye, *ceyh;
	double *ez, *ceze, *cezh;
	double *ezright, *ezleft, *eztop, *ezbottom;
	float *ez_float;
	int M, N, sizeZ;
	int time, maxTime;
	int type, nCells;
	double cdtds, dx, dt;
	double src_f, lambda, N_lambda;
	double DOMX, DOMY;				// Physical domain size in meters
	double src_x, src_y;			// Physical location of source
	int src_i, src_j, src_type;		// Array location of source and source type
};

typedef struct Grid Grid;


void gridInit(Grid *g);


#endif
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
	float *ez_float;
	int sizeX, sizeY, sizeZ;
	int time, maxTime;
	int type;
	double cdtds;
};

typedef struct Grid Grid;


void gridInit(Grid *g);


#endif
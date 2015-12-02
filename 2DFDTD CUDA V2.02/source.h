
#ifndef _SOURCE_H
#define _SOURCE_H

//#include <math.h>
//#include <stdio.h>
//#include <stdlib.h>
#include "grid_2d.h"

void ezIncInit(Grid *g);
double ezIncRicker(int time, double location);
double ezIncCos(double time);
void snapshotInit2d(Grid *g);
void snapshot2d(Grid *g);
void abcInit(Grid *g);
void abc(Grid *g);
void tfsfInit(Grid *g);
void tfsfUpdate(Grid *g);


#endif
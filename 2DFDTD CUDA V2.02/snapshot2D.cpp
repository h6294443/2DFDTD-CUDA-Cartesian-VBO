/*	Function to record the 2D field to a file.  The data is stored as binary (raw) data.	*/
/*	This source file belongs to project "Ch8_8.4 (TMz Example)	*/

#include <stdio.h>
#include <stdlib.h>
#include "grid_2d.h"

static int temporalStride = -2, frame = 0, startTime,
startNodeX, endNodeX, spatialStrideX,
startNodeY, endNodeY, spatialStrideY;
static char basename[80];

void snapshotInit2d(Grid *g) {

	int choice = 1;

	//printf("Do you want 2D snapshots? (1=yes, 0=no) ");
	//scanf_s("%d", &choice);
	if (choice == 0) {
		temporalStride = -1;
		return;
	}

	//printf("duration of simulation is %d steps.\n", g->maxTime);
	//printf("Enter start time and temporal stride: ");
	//scanf_s(" %d %d", &startTime, &temporalStride);
	startTime = 0;
	temporalStride = 1;

	//printf("In the x-direction, the grid has %d total nodes"
	//	" (ranging from 0 to %d)\n", g->sizeX, g->sizeX - 1);
	//printf("Enter first node, last node, and spatial stride: ");
	//scanf_s(" %d %d %d", &startNodeX, &endNodeX, &spatialStrideX);
	startNodeX = 0; endNodeX = g->sizeX - 1; spatialStrideX = 1;

	//printf("In the y-direction, the grid has %d total nodes"
//		" (ranging from 0 to %d)\n", g->sizeY, g->sizeY - 1);
	//printf("Enter first node, last node, and spatial stride: ");
	//scanf_s(" %d %d %d", &startNodeY, &endNodeY, &spatialStrideY);
	startNodeY = 0; endNodeY = g->sizeY - 1; spatialStrideY = 1;
	//printf("Enter the base name: ");
	//scanf_s(" %c", basename);
	*basename = 's';

	return;
}

void snapshot2d(Grid *g) {
	int mm, nn;
	double dim1, dim2, temp;
	char filename[100];
	FILE *out;

	/*	Ensure temporal stride is a reasonable value	*/
	if (temporalStride == -1) {
		return;
	} if (temporalStride < -1) {
		fprintf(stderr,
			"snapshot2d: snapshotInit2d must be called before snapshot.\n"
			"			 Temporal stride must be set to positive value.\n");
		exit(-1);
	}

	/*	Get snapshot if temporal conditions met	*/
	if (g->time >= startTime && (g->time - startTime) % temporalStride == 0) {
		sprintf_s(filename, "%s.%d", basename, frame++);
		out = fopen(filename, "wb");

		/*	Write dimensions to output file -- express dimensions as floats	*/
		dim2 = (endNodeX - startNodeX) / spatialStrideX + 1;
		dim1 = (endNodeY - startNodeY) / spatialStrideY + 1;
		fwrite(&dim1, sizeof(double), 1, out);
		fwrite(&dim2, sizeof(double), 1, out);

		/*	Write remaining data	*/
		for (nn = endNodeY; nn >= startNodeY; nn -= spatialStrideY)
			for (mm = startNodeX; mm <= endNodeX; mm += spatialStrideX) {
				temp = g->ez[mm + (nn*g->sizeX)];									//	This stores the data as a float through type-casting
				fwrite(&temp, sizeof(double), 1, out);						//	Actually write the float
			}

		fclose(out);	//close file
	}

	return;
}


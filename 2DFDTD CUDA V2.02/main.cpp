// ____________________________________________________________________	//
// V2.02 uses a vertex buffer object (vbo) instead of a pixel buffer    //
// object as before.  This is done to bring consistency to the way both //
// Cartesian and Hex grids are displayed.								//
// ____________________________________________________________________	//
// All fields get initialized on host memory in the						//
// Grid structure.  However, they are no longer copied to and from the  //
// device each time iteration.  Instead, they are passed to device      //
// pointers at the start of the simulation and remain there. The vastly //
// reduced memory transfers improve performance tremendously.			//
// ____________________________________________________________________	//
// At each glutMainLoop() iteration, exactly one field update is per-   //
// formed (one time step).  The Ez field is generated in both a double  //
// and a float variable.  The float array is then passed to the image   //
// handling functions.  The resulting pbo is applied to a full-screen   //
// quad as a texture.													//
// ____________________________________________________________________	//
// The grid and device pointers are extern variables first declared in  //
// global.h, then initialized elsewhere.  global.h is included every-   //
// where it is needed, ensuring global variables and valid device poin- //
// ters that can be read from/written to by all parts of the program.   //
// Note that device functions cannot access host variables.				//
// ____________________________________________________________________	//
// Similarly, host functions cannot access device pointers directly.    //
// This makes the source update at each time iteration difficult.  The  //
// solution so far is to perform the source update inside the Ez Update //
// kernel itself.  Time for the cosine source function is passed as an  //
// integer originating from g->time.  Immediately after the kernel call //
// returns control to the host, g->time is increased by one, ensuring   //
// progression of the time steps even without an explicit loop to do so.//
// ____________________________________________________________________	//
// V2.02 also includes full field coefficient arrays Chxe, Chxh, Chye,  //
// Chyh, Cezh, Ceze.  Performance should be nearly identical because    //
// the fields are copied only once at the beginning of the simulation   //
// and one more time at the end of the simulation.						//
// ____________________________________________________________________	//
// Usage:  Specify simulation parameters in parameters.h, then compile. //
// ____________________________________________________________________	//
/* Matt Engels, October 25, 2015*/
#include "global.h"
#include "grid_2d.h"
#include "cuda_functions.h"
#include "source.h"
#include "graphics.h"
//#include <time.h>
//#include <Windows.h>
//#include "parameters.h"
// CARTESIAN

int main(int argc, char** argv)
{
	gridInit(g);							// Initialize the grid
	ezIncInit(g);							// Initialize the sources
	initializeGlobalDevicePointers();		// Initialize all global dev pointers to zero
	runFdtdWithFieldDisplayVbo(argc, argv);
	//runFdtdWithFieldDisplay(argc, argv);
		
}


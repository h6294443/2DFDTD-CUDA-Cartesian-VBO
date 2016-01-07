#include "parameters.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// The following are CUDA-related function prototypes
// Full functions are in cuda_functions.cu
// Standard functions first
void update_all_fields_CUDA();			// Updates all fields
void resetBeforeExit();					// Required CUDA calls after function exit
void deviceSyncAfterKernelLaunch();		// Required CUDA calls after kernel call
void pickGPU(int gpuid);				// Picks a specific GPU in a multi-GPU system.  Starts at 0.
void checkErrorAfterKernelLaunch();		// CUDA error checking after a kernel was launched
void initializeGlobalDevicePointers();	// Initializes the GPU memory ("Device") array pointers
int copyTMzArraysToDevice();			// Copies the host-side arrays to the device (CPU->GPU)
bool copyFieldSnapshotsFromDevice();	// Copies field snapshots from device to host (GPU->CPU)
bool deallocateCudaArrays();			// used to cudaFree() all device arrays

// Now the CUDA kernels
__global__ void HxHyUpdate_Kernel(double *dHx, double *dChxh, double *dChxe, double *dHy, double *dChyh, double *dChye, double *dEz, int M, int N);
__global__ void EzUpdate_Kernel(double *dEz, double *dCezh, double *dCeze, float *dImEz, double *dHx, double *dHy, int M, int N);
__global__ void Source_Update_Kernel(double *dEz, float *dImEz, int x, int y, int type, int time, double factor, int loc, double ppw, double Sc, int start_time, int stop_time, int width);


// The following are OpenGL/CUDA Interop-related function prototypes
// Full functions are in cuda-opengl_functions.cu
// Standard functions first
void createImageOnGpuVbo();				// Creates the color buffer from Ez field values
void create_Grid_points_only(float4* dDptr, float *dEz_float);	// Creates the grid geometry

// Now the CUDA kernels
__global__ void create_Grid_points_only_kernel(float4 *dDptr, float *dEz_float, int width, int height, float delta_x, float delta_y);
__global__ void find_min_and_max_on_gpu(int nblocks, float* field, float* minimum_field_value, float* maximum_field_value);
__global__ void create_image_on_gpu_vbo_kernel(uchar4 *colorPos, float4 *pos, float* Ez, int M, float minval, float maxval, int time, int width, int height);


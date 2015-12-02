#include "global.h"
#include "cuda_functions.h"
#include "source.h"
#include <math.h>

__global__ void Source_Update_Kernel(double *dEz, float *dImEz, int x, int y, int type, int time, double factor, int loc, double ppw, double Sc, int start_time, int stop_time)
{
	// Map from threadIdx/blockIdx to cell position
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = row * blockDim.x *gridDim.x + col;
	if (time > start_time && time < stop_time) {
		if ((col == src_pos_x) && (row == src_pos_y)) {
			if (type == 0) {        // Cosine
				dEz[offset] = dEz[offset] + cos(2 * PI*factor * time);
				dImEz[offset] = __double2float_rd(dEz[offset]);
			}
			else if (type == 1) {   // Sine
				dEz[offset] = dEz[offset] + sin(2 * PI*factor * time);
				dImEz[offset] = __double2float_rd(dEz[offset]);
			}
			else if (type == 2) {   // Ricker Wavelet
				double fraction = PI*(Sc * time - loc) / (ppw - 1.0);
				dEz[offset] = dEz[offset] + fraction * fraction;
				dImEz[offset] = __double2float_rd(dEz[offset]);
			}
		}
	}
}

__global__ void HxHyUpdate_Kernel(double *dHx, double *dChxh, double *dChxe, double *dHy, double *dChyh, double *dChye, double *dEz, int M, int N)
{
	// Map from threadIdx/blockIdx to cell position
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = row * blockDim.x *gridDim.x + col;

	int size_Hx = M * (N - 1);
	int size_Hy = (M - 1) * N;
	int top = offset + blockDim.x * gridDim.x;
	int right = offset + 1;

	////////////////////////////////////////////////////////////////////////////////////
	// Calculate Hx
	if ((row == M - 1)) top -= M;
	if (offset < size_Hx) {
		dHx[offset] = dChxh[offset] * dHx[offset] - dChxe[offset] * (dEz[top] - dEz[offset]);
	}
	__syncthreads();								// only, not actual errors

	////////////////////////////////////////////////////////////////////////////////////
	// Calculate Hy
	if ((col == M - 1) || (col == M - 2)) right--;
	if (offset < size_Hy) {
		dHy[offset] = dChyh[offset] * dHy[offset] + dChye[offset] * (dEz[right] - dEz[offset]);		
	}
	__syncthreads();
}

__global__ void EzUpdate2D_Kernel(double *dEz, double *dCezh, double *dCeze, float *dImEz, double *dHx, double *dHy, int DIM)
{
	// Map from threadIdx/blockIdx to cell position
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = row * blockDim.x *gridDim.x + col;
	double Sc = 1 / ((double)sqrt(2.0));

	int total = DIM*DIM;
	int left = offset - 1;
	int right = offset + 1;
	int top = offset + blockDim.x * gridDim.x;
	int bottom = offset - blockDim.x * gridDim.x;
		
	if (col == 0)			left++;
	if (col == DIM - 1)		right--;
	if ((row == DIM - 1))	top -= DIM;
	if (row == 0)			bottom += DIM;

	if ((col == 0) || (col == (M - 1)) || (row == 0) || (row == (N - 1)))	dEz[offset] = 0.0;
	//else if ((col == src_pos_x) && (row == src_pos_y))	dEz[offset] = cos(2 * PI*factor * time);
	//else if ((col == src_pos_x2) && (row == src_pos_y2)) dEz[offset] = cos(2 * PI * time / 200);
	else {
		if (offset < total)
			dEz[offset] = dCeze[offset] * dEz[offset] +
			dCezh[offset] * ((dHy[offset] - dHy[left]) - (dHx[offset] - dHx[bottom]));
		dImEz[offset] = __double2float_rd(dEz[offset]);				// Populate the image data Ez array
	}
}

void update_all_fields_CUDA()
{
	// Calculate CUDA grid dimensions.  Block dimension fixed at 32x32 threads
	int Bx = (g->sizeX + (TILE_SIZE - 1)) / TILE_SIZE;
	int By = (g->sizeY + (TILE_SIZE - 1)) / TILE_SIZE;
	dim3 BLK(Bx, By, 1);
	dim3 THD(TILE_SIZE, TILE_SIZE, 1);

	double factor = Sc / N_lambda;
	
		
	// Launch kernel to update TMz magnetic field components Hx and Hy and time the whole thing
	// Note: timing is for debugging and performance measurement only
	//cudaEvent_t start, stop;							// Declare the start and stop events
	//cudaEventCreate(&start);							// Create the start event
	//cudaEventCreate(&stop);								// Create the stop event
	//cudaEventRecord(start);								// Start timer
	
	HxHyUpdate_Kernel << <BLK, THD >> >(dev_hx, dev_chxh, dev_chxe, dev_hy, dev_chyh, dev_chye, dev_ez, g->sizeX, g->sizeY);
	
	//cudaEventRecord(stop);								// Stop timer

	checkErrorAfterKernelLaunch();						// Check for any errors launching the kernel
	deviceSyncAfterKernelLaunch();						// Do a device sync 
		
	//copyTMzMagneticFieldsToHost(g, dev_hx, dev_hy);	// Copy magnetic TMz fields from device (GPU) back to host (CPU)
	
	//cudaEventSynchronize(stop);							// Make sure events are synchronized for accurate time recording
	//float milliseconds = 0;								// Timer variable in milli-seconds
	//cudaEventElapsedTime(&milliseconds, start, stop);	// Calculates the ellapsed time between the properly synchronized start and stop events
	//printf("Elapsed time for HxHyUpdate_Kernel, one iteration: %f.\n", milliseconds);

	// Launch kernel to update Ez field component and time the whole thing
	// Note: timing is for debugging and performance measurement only
	//cudaEventRecord(start);								// Start timer
	//dev_ez[10] = cos(2 * PI*g->time / 25);
	EzUpdate2D_Kernel << <BLK, THD >> >(dev_ez, dev_cezh, dev_ceze, dev_ez_float, dev_hx, dev_hy, g->sizeX);
	
	//cudaEventRecord(stop);								// Stop timer
	
	checkErrorAfterKernelLaunch();						// Check for any errors launching the kernel
	deviceSyncAfterKernelLaunch();						// Do a device sync 
	
	Source_Update_Kernel << <BLK, THD >> >(dev_ez, dev_ez_float, src_pos_x, src_pos_y, 2, g->time, factor, 150, N_lambda, Sc, 0, 1500);
	
	//copyTMzElectricFieldsToHost(g, dev_ez);			// Copy electric field component from device to host

	//cudaEventSynchronize(stop);							// Make sure events are synchronized for accurate time recording
	//cudaEventElapsedTime(&milliseconds, start, stop);	// Calculates the ellapsed time between the properly synchronized start and stop events
	//printf("Elapsed time for EzUpdate2D_Kernel, one iteration: %f.\n", milliseconds);

	//freeTMzFieldsOnDevice(dev_hx, dev_hx_float, dev_hy, dev_hy_float, dev_ez, dev_ez_float);		// Free up the device (GPU) memory
		
	g->time += 1;										// Must advance time manually here	
}


void resetBeforeExit() {

	cudaError_t cudaStatus;
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		//return 1;
	}
	
}

void pickGPU(int gpuid) {

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(gpuid);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
}

void checkErrorAfterKernelLaunch() {
	
	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
}

void deviceSyncAfterKernelLaunch() {

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
}

void initializeGlobalDevicePointers() {
	
	// Initialize the extern variables below prior to first use
	dev_hx = 0;							// The double-precision Hx field on Device memory
	dev_ez = 0;							// Same for Ez
	dev_hy = 0;							// Same for Hy
	dev_chxh = 0;
	dev_chxe = 0;
	dev_chyh = 0;
	dev_chye = 0;
	dev_cezh = 0;
	dev_ceze = 0;
	dev_ez_float = 0;					// The single-precision fields on Device memory, 
	dev_hx_float = 0;					// used as OpenGL interop buffer 
	dev_hy_float = 0;
}

int copyTMzArraysToDevice()
{
	int hxsize = g->sizeX * (g->sizeY - 1);
	int ezsize = g->sizeX * g->sizeY;
	int hysize = (g->sizeX - 1) * g->sizeY;
	
	//int size_i = sizeof(int);
	//int size_c = sizeof(char);
	int size_f = sizeof(float);
	int size_d = sizeof(double);
	
	cudaError_t et;

	et = cudaMalloc((void**)&dev_hx,	   hxsize*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_chxh,	   hxsize*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_chxe,	   hxsize*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_hy,	   hysize*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_chyh, 	   hysize*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_chye,	   hysize*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_ez,	   ezsize*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_cezh,	   ezsize*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_ceze,	   ezsize*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_hx_float, hxsize*size_f);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_hy_float, hysize*size_f);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_ez_float, ezsize*size_f);	if (et == cudaErrorMemoryAllocation) return 1;

	// Note that the float copies of the field components do not need to be copied because
	// they are generated by the update kernel.
	cudaMemcpy(dev_hx,		g->hx,		hxsize*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_chxh,	g->chxh,	hxsize*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_chxe,	g->chxe,	hxsize*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_hy,		g->hy,		hysize*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_chyh,	g->chyh,	hysize*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_chye,	g->chye,	hysize*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ez,		g->ez,		ezsize*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cezh,	g->cezh,	ezsize*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ceze,	g->ceze,	ezsize*size_d,	cudaMemcpyHostToDevice);
	
	et = cudaMalloc((void**)&dvminimum_field_value, sizeof(float)*TILE_SIZE);	if (et == cudaErrorMemoryAllocation) return 1;	
	et = cudaMalloc((void**)&dvmaximum_field_value, sizeof(float)*TILE_SIZE);	if (et == cudaErrorMemoryAllocation) return 1;	

	return 0;
}

bool copyFieldSnapshotsFromDevice()
{
	int hxsize = g->sizeX * (g->sizeY - 1);
	int ezsize = g->sizeX * g->sizeY;
	int hysize = (g->sizeX - 1) * g->sizeY;
	int size_d = sizeof(double);
	int size_f = sizeof(float);		// only for debugging use 

	// Copy an electric field frame.
	cudaMemcpy(g->ez, dev_ez, ezsize * size_d, cudaMemcpyDeviceToHost);
	cudaMemcpy(g->hx, dev_hx, hxsize * size_d, cudaMemcpyDeviceToHost);
	cudaMemcpy(g->hy, dev_hy, hysize * size_d, cudaMemcpyDeviceToHost);
	cudaMemcpy(g->ez_float, dev_ez_float, ezsize *size_f, cudaMemcpyDeviceToHost);

	/*for (int i=0; i < (g->sizeX*g->sizeY); i++){
		if (g->ez_float[i] > 0 || g->ez_float[i] < 0) {
			printf("g->ez_float[%i] = %f\n", i, g->ez_float[i]);
		}
	}*/

	return true;
}

bool deallocateCudaArrays()
{
	cudaFree(dev_hx);
	cudaFree(dev_chxh);
	cudaFree(dev_chxe);
	cudaFree(dev_hy);
	cudaFree(dev_chyh);
	cudaFree(dev_chye);
	cudaFree(dev_ez);
	cudaFree(dev_cezh);
	cudaFree(dev_ceze);
	//cudaFree(dev_hx_float);
	//cudaFree(dev_hy_float);
	cudaFree(dev_ez_float);
	cudaFree(dvminimum_field_value);
	cudaFree(dvmaximum_field_value);

	return true;
}
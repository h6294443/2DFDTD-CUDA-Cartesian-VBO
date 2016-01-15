#include "global.h"
#include "cuda_functions.h"
#include "source.h"
#include <math.h>

__global__ void Source_Update_Kernel(double *dEz, float *dImEz, int x, int y, int type, int time, double factor, int loc, double ppw, double Sc, int start_time, int stop_time, int width)
{
	// Map from threadIdx/blockIdx to cell position
	int offset = blockIdx.x * blockDim.x + threadIdx.x;	// Calculate linear offset for 1-D unrolled array
	int j = offset / width;							// Creates a virtual row index for the 1-D case, needed for odd/even row check
	int i = offset % width;							// Keeping both offset and i for clarity
		
	if (time > start_time && time < stop_time) {
		if ((i == x) && (j == y)) {
			if (type == 0) {        // Cosine
				dEz[offset] = dEz[offset] + 10* cos(2 * PI*factor * time);
				dImEz[offset] = __double2float_rd(dEz[offset]);
			}
			else if (type == 1) {   // Sine
				dEz[offset] = dEz[offset] + 10*sin(2 * PI*factor * time);
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
	int offset = blockIdx.x * blockDim.x + threadIdx.x;	// Calculate linear offset for 1-D unrolled array
	int j = offset / M;							// Creates a virtual row index for the 1-D case, needed for odd/even row check
	int i = offset % M;							// Keeping both offset and i for clarity
	int size_Hx = M * (N - 1);
	int size_Hy = (M - 1) * N;
	int top = offset + M;
	int right = offset + 1;

	////////////////////////////////////////////////////////////////////////////////////
	// Calculate Hx
	if ((j == N - 1)) top -= M;
	if (offset < size_Hx) {
		dHx[offset] = dChxh[offset] * dHx[offset] - dChxe[offset] * (dEz[top] - dEz[offset]);
	}
	__syncthreads();								// only, not actual errors

	////////////////////////////////////////////////////////////////////////////////////
	// Calculate Hy
	if ((i == M - 1) || (i == M - 2)) right--;
	if (offset < size_Hy) {
		dHy[offset] = dChyh[offset] * dHy[offset] + dChye[offset] * (dEz[right] - dEz[offset]);		
	}
	__syncthreads();
}

__global__ void EzUpdate_Kernel(double *dEz, double *dCezh, double *dCeze, float *dImEz, double *dHx, double *dHy, int M, int N)
{
	// Map from threadIdx/blockIdx to cell position
	int offset = blockIdx.x * blockDim.x + threadIdx.x;	// Calculate linear offset for 1-D unrolled array
	int j = offset / M;							// Creates a virtual row index for the 1-D case, needed for odd/even row check
	int i = offset % M;							// Keeping both offset and i for clarity
	
	int total = M*N;
	int left = offset - 1;
	int right = offset + 1;
	int top = offset + M;
	int bottom = offset - M;
		
	if (i == 0)			left++;
	if (i == M - 1)	right--;
	if ((j == N - 1))	top -= M;
	if (j == 0)			bottom += M;

	if ((i == 0) /*|| (i == (M - 1)) */|| (j == 0) || (j == (N - 1)))	dEz[offset] = 0.0;

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
	int TILE_SQUARED = TILE_SIZE * TILE_SIZE;
	int Bx = (TILE_SQUARED - 1 + g->nCells) / TILE_SQUARED;
	dim3 BLK(Bx, 1, 1);									// Grid-block dimension for the 1-D case
	dim3 THD(TILE_SQUARED, 1, 1);							// Thread-block dimension for the 1-D case
	double factor = g->cdtds / g->N_lambda;

	HxHyUpdate_Kernel << <BLK, THD >> >(dev_hx, dev_chxh, dev_chxe, dev_hy, dev_chyh, dev_chye, dev_ez, g->M, g->N);
	checkErrorAfterKernelLaunch();						// Check for any errors launching the kernel
	deviceSyncAfterKernelLaunch();						// Do a device sync 
		
	//copyTMzMagneticFieldsToHost(g, dev_hx, dev_hy);	// Copy magnetic TMz fields from device (GPU) back to host (CPU)
	EzUpdate_Kernel << <BLK, THD >> >(dev_ez, dev_cezh, dev_ceze, dev_ez_float, dev_hx, dev_hy, g->M, g->N);
	checkErrorAfterKernelLaunch();						// Check for any errors launching the kernel
	deviceSyncAfterKernelLaunch();						// Do a device sync 
	
	abc(g);

	Source_Update_Kernel << <BLK, THD >> >(dev_ez, dev_ez_float, g->src_i, g->src_j, g->src_type, g->time, factor, 150, g->N_lambda, g->cdtds, 0, 1500, g->M);
	checkErrorAfterKernelLaunch();						// Check for any errors launching the kernel
	deviceSyncAfterKernelLaunch();						// Do a device sync 
	g->time += 1;
	//copyTMzElectricFieldsToHost(g, dev_ez);			// Copy electric field component from device to host
	//freeTMzFieldsOnDevice(dev_hx, dev_hx_float, dev_hy, dev_hy_float, dev_ez, dev_ez_float);		// Free up the device (GPU) memory		
}

void resetBeforeExit() {
	cudaError_t cudaStatus;
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceReset failed!");	
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
	if (cudaStatus != cudaSuccess) 
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
}

void initializeGlobalDevicePointers() {
	// Initialize the extern variables below prior to first use
	dev_hx = 0;					// The double-precision Hx field on Device memory
	dev_ez = 0;					// Same for Ez
	dev_hy = 0;					// Same for Hy
	dev_chxh = 0;
	dev_chxe = 0;
	dev_chyh = 0;
	dev_chye = 0;
	dev_cezh = 0;
	dev_ceze = 0;
	dev_ez_float = 0;			// The single-precision fields on Device memory, 
	dev_hx_float = 0;			// used as OpenGL interop buffer 
	dev_hy_float = 0;

	dev_ezright = 0;			// Right Ez field border, for the 2nd Order Mur ABC
	dev_ezleft = 0;
	dev_eztop = 0;
	dev_ezbottom = 0;
}

int copyTMzArraysToDevice() {
	int hxsize = g->M * (g->N - 1);
	int ezsize = g->M * g->N;
	int hysize = (g->M - 1) * g->N;
	int size_f = sizeof(float);
	int size_d = sizeof(double);
	cudaError_t et;

	et = cudaMalloc((void**)&dev_hx,	   hxsize*size_d);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_chxh,	   hxsize*size_d);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_chxe,	   hxsize*size_d);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_hy,	   hysize*size_d);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_chyh, 	   hysize*size_d);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_chye,	   hysize*size_d);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_ez,	   ezsize*size_d);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_cezh,	   ezsize*size_d);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_ceze,	   ezsize*size_d);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_hx_float, hxsize*size_f);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_hy_float, hysize*size_f);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_ez_float, ezsize*size_f);	if (et == cudaErrorMemoryAllocation) return 1;
	
	et = cudaMalloc((void**)&dev_ezright,  6*g->N*size_d);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_ezleft,   6*g->N*size_d);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_eztop,    6*g->M*size_d);	if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_ezbottom, 6*g->M*size_d);	if (et == cudaErrorMemoryAllocation) return 1;

	// Note that the float copies of the field components do not need to be copied because
	// they are generated by the update kernel.
	cudaMemcpy(dev_hx,		g->hx,		hxsize*size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_chxh,	g->chxh,	hxsize*size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_chxe,	g->chxe,	hxsize*size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_hy,		g->hy,		hysize*size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_chyh,	g->chyh,	hysize*size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_chye,	g->chye,	hysize*size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ez,		g->ez,		ezsize*size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cezh,	g->cezh,	ezsize*size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ceze,	g->ceze,	ezsize*size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ezright, g->ezright, 6*g->N*size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ezleft,  g->ezleft,  6*g->N*size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ezbottom,g->ezbottom,6*g->M*size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_eztop,   g->eztop,	6*g->M*size_d, cudaMemcpyHostToDevice);

	et = cudaMalloc((void**)&dvminimum_field_value, sizeof(float)*TILE_SIZE);	if (et == cudaErrorMemoryAllocation) return 1;	
	et = cudaMalloc((void**)&dvmaximum_field_value, sizeof(float)*TILE_SIZE);	if (et == cudaErrorMemoryAllocation) return 1;	
	return 0;
}

bool copyFieldSnapshotsFromDevice() {
	int hxsize = g->M * (g->N - 1);
	int ezsize = g->M * g->N;
	int hysize = (g->M - 1) * g->N;
	int size_d = sizeof(double);
	int size_f = sizeof(float);		// only for debugging use 

	// Copy an electric field frame.
	cudaMemcpy(g->ez, dev_ez, ezsize * size_d, cudaMemcpyDeviceToHost);
	cudaMemcpy(g->hx, dev_hx, hxsize * size_d, cudaMemcpyDeviceToHost);
	cudaMemcpy(g->hy, dev_hy, hysize * size_d, cudaMemcpyDeviceToHost);
	cudaMemcpy(g->ez_float, dev_ez_float, ezsize *size_f, cudaMemcpyDeviceToHost);
	
	return true;
}

bool deallocateCudaArrays() {
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
	cudaFree(dev_ezright);
	cudaFree(dev_ezleft);
	cudaFree(dev_ezbottom);
	cudaFree(dev_eztop);
	cudaFree(dvminimum_field_value);
	cudaFree(dvmaximum_field_value);

	return true;
}
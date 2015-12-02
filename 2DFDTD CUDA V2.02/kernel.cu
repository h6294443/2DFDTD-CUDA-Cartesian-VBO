
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "parameters.h"
#include "grid-2d.h"
#include <stdio.h>



//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t addWithCuda(Grid *g);

__global__ void addKernel(double *c, double *a, double *b)
{
	int i = threadIdx.x;
	int offset = 1;
	double temp = c[i];
	if (i == 9) 
		offset = 0;
	c[i] = a[i] + b[i] + temp;
	__syncthreads();

}

int main()
{
	cudaError_t cudaStatus;
	Grid *g = new Grid;
	gridInit(g);
	const int arraySize = g->sizeX*(g->sizeY-1);
	

for (int i = 0; i < 10; i++) {
	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(g);
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "addWithCuda failed!");
	return 1;
	}
	for (int j = 0; j<arraySize-1;j++) 
		printf("%g + %g = %g\n", g->chxh[j], g->chxe[j], g->hx[j]);
	//printf("test\n");
		
}
	
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
cudaError_t addWithCuda(Grid *g)
{
    double *dev_chxh = 0;
    double *dev_chxe = 0;
    double *dev_hx = 0;
	cudaError_t cudaStatus;
	int size = g->sizeX * (g->sizeY - 1);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_hx, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	
	cudaStatus = cudaMalloc((void**)&dev_chxh, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_chxe, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_chxh, g->chxh, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_hx, g->hx, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_chxe, g->chxe, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_hx, dev_chxh, dev_chxe);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(g->hx, dev_hx, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_hx);
	cudaFree(dev_chxh);
    cudaFree(dev_chxe);
    
    return cudaStatus;
}

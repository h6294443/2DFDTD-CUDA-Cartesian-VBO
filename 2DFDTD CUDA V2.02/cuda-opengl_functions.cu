#include "graphics.h"

texture<float4, 2, cudaReadModeElementType> inTex;
__constant__ unsigned int dvrgb[256];

__global__ void create_Grid_points_only_kernel(float4 *dDptr, float *dEz_float, int width, int height, float delta_x, float delta_y) {
	// This kernel turns the blank vertex spatial array into a properly formatted 
	// array of vertices that can be drawn during the display call.  As the spatial
	// information does not change during the simulation, this kernel gets called
	// only once.
	__shared__ float dev_delta_x;
	__shared__ float dev_delta_y;
	dev_delta_x = delta_x;								// x-resolution
	dev_delta_y = delta_y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = j * blockDim.x * gridDim.x + i;					// (i, j)
	float x, y;

	// calculate uv coordinates
	float u = i / (float)width;
	float v = j / (float)height;
	u = u*2.0f - 1.0f;
	v = v*2.0f - 1.0f;
	float w = 0.0f;

	x = -1.0 + i * dev_delta_x;
	y = -1.0 + j * dev_delta_y;
	u = x;
	w = y;
	v = 0.0f;// 1e-1*dEz_float[offset]; // 0.0f;

	// write output vertex
	if (offset < (width*height)) dDptr[offset] = make_float4(u, w, v, 1.0f);
}

__global__ void find_min_and_max_on_gpu(int nblocks, float* field, 
										float* minimum_field_value, 
										float* maximum_field_value)
{
	__shared__ float minarr[1024];
	__shared__ float maxarr[1024];

	int i = threadIdx.x;
	int nTotalThreads = blockDim.x;

	minarr[i] = field[i];
	maxarr[i] = minarr[i];
	for (int j = 1; j<nblocks; j++)
	{
		minarr[i + nTotalThreads] = field[i + nTotalThreads*j];
		if (minarr[i] > minarr[i + nTotalThreads])
			minarr[i] = minarr[i + nTotalThreads];

		if (maxarr[i] < minarr[i + nTotalThreads])
			maxarr[i] = minarr[i + nTotalThreads];
		__syncthreads();
	}
	__syncthreads();

	while (nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		if (threadIdx.x < halfPoint)
		{
			float temp = minarr[i + halfPoint];

			if (temp < minarr[i]) minarr[i] = temp;

			temp = maxarr[i + halfPoint];
			if (temp > maxarr[i]) maxarr[i] = temp;
		}
		__syncthreads();
		nTotalThreads = (nTotalThreads >> 1);
	}
	if (i == 0)
	{
		minimum_field_value[0] = minarr[0];
		maximum_field_value[0] = maxarr[0];
	}
}

void createColormapOnGpu()
{
	cudaError_t et;
	et = cudaMemcpyToSymbol(dvrgb, rgb, 256 * sizeof(int), 0, cudaMemcpyHostToDevice);
}

__global__ void
create_image_on_gpu(unsigned int* g_odata, float* Ez, int M, float minval, float maxval)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int cind;
	float F;

	int ci = j*M + i;
	int ti = (j + 1)*M + i;
	if (j == M - 1) ti = (j)*M + i;
	F = Ez[ti] - minval;
	cind = floor(255 * F / (maxval - minval));
	if (cind > 255) cind = 255;
	g_odata[ci] = dvrgb[cind];
}

__global__ void create_image_on_gpu_vbo_kernel(uchar4 *colorPos, float4 *pos, float* Ez, int M, float minval, float maxval, int time, int width, int height)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float F;
	float temp;
	temp = minval;

	int offset = j*M + i;
	//int ti = (j + 1)*M + i;
	//if (j == M - 1) ti = (j)*M + i;
	//F = Ez[ti] - minval;
	//cind = floor(255 * F / (maxval - minval));
	//if (cind > 255) cind = 255;
	//g_odata[ci] = dvrgb[cind];

	F = Ez[offset] - minval;
	F = F / (maxval - minval);
	
	// calculate uv coordinates
	float u = i / (float)width;
	float v = j / (float)height;
	u = u*2.0f - 1.0f;
	v = v*2.0f - 1.0f;

	// calculate simple sine wave pattern
	float freq = 4.0f;
	float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	colorPos[offset].w = 0;
	colorPos[offset].x = 255.f *0.7*(F);
	colorPos[offset].y = 255.f *0.3*(F);
	colorPos[offset].z = 255.f *0.5*(F);

	// write output vertex
	pos[offset] = make_float4(u, w, v, 1.0f);
	//colorPos[offset].w = 0;
	//colorPos[offset].x = 255.f *0.5*(1.f + sinf(w + i));
	//colorPos[offset].y = 255.f *0.5*(1.f + sinf(i)*cosf(j));
	//colorPos[offset].z = 255.f *0.5*(1.f + sinf(w + time / 10.f));

}


void createImageOnGpuVbo()	// argument g_odata is the float Ez field 
{												// array, coming in as a device pointer
	dim3 block(TILE_SIZE, TILE_SIZE, 1);
	dim3 grid(M / block.x, N / block.y, 1);
	dim3 gridm = dim3(1, 1, 1);
	dim3 blockm = dim3(TILE_SIZE*TILE_SIZE, 1, 1);
	int  nblocks = grid.x * grid.y;
	float minval;
	float maxval;
	float *dvF;

	//if (show_Ez) dvF = dev_ez_float; else dvF = dev_hx_float;
	dvF = dev_ez_float;

	find_min_and_max_on_gpu << < gridm, blockm >> >(nblocks, dvF, dvminimum_field_value, dvmaximum_field_value);

	cudaMemcpy(&minval, dvminimum_field_value, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&maxval, dvmaximum_field_value, sizeof(float), cudaMemcpyDeviceToHost);

	if (minval > 0.0) minval = 0.0;
	if (maxval < 0.0) maxval = 0.0;
	if (abs(minval) > maxval) maxval = -minval; else minval = -maxval;
	if (minval < global_min_field) global_min_field = minval;
	if (maxval > global_max_field) global_max_field = maxval;

	//cudaMemcpy(g->ez_float, dvF, g->nCells*sizeof(float), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < g->nCells; i++)
	//	if (g->ez_float[i] > 0.001f || g->ez_float[i] < -0.001f)
	//		printf("ez_float[%i] = %f\n", i, g->ez_float[i]);

	//minval = -1.0;	maxval = 1.0;	global_min_field = -1.0; global_max_field = 1.0;
	//the following kernel now takes a uchar4 array, not uint
	create_image_on_gpu_vbo_kernel << < grid, block >> >(cptr, dptr, dvF, M, global_min_field, global_max_field, g->time, g->sizeX, g->sizeY);
}

void createImageOnGpu(unsigned int* g_odata)
{
	dim3 block(TILE_SIZE, TILE_SIZE, 1);
	dim3 grid(M / block.x, N / block.y, 1);
	dim3 gridm = dim3(1, 1, 1);
	dim3 blockm = dim3(TILE_SIZE*TILE_SIZE, 1, 1);
	int  nblocks = grid.x * grid.y;
	float minval;
	float maxval;
	float *dvF;

	//if (show_Ez) dvF = dev_ez_float; else dvF = dev_hx_float;
	dvF = dev_ez_float;

	find_min_and_max_on_gpu << < gridm, blockm >> >(nblocks, dvF, dvminimum_field_value, dvmaximum_field_value);

	cudaMemcpy(&minval, dvminimum_field_value, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&maxval, dvmaximum_field_value, sizeof(float), cudaMemcpyDeviceToHost);

	if (minval>0.0) minval = 0.0;
	if (maxval<0.0) maxval = 0.0;
	if (abs(minval)>maxval) maxval = -minval; else minval = -maxval;
	if (minval<global_min_field) global_min_field = minval;
	if (maxval>global_max_field) global_max_field = maxval;

	//minval = -1.0;	maxval = 1.0;	global_min_field = -1.0; global_max_field = 1.0;
	
	create_image_on_gpu << < grid, block >> >(g_odata, dvF, M, global_min_field, global_max_field);
}

void create_Grid_points_only(float4* dDptr, float *dEz_float)
{
	// This function and kernel get called only once to create the spatial portion
	// of the vertex buffer object.  The colors will be updated seperately each loop.
	// Calculate CUDA grid dimensions.  Block dimension fixed at 32x32 threads
	int Bx = (g->sizeX + (TILE_SIZE - 1)) / TILE_SIZE;
	int By = (g->sizeY + (TILE_SIZE - 1)) / TILE_SIZE;
	dim3 BLK(Bx, By, 1);
	dim3 THD(TILE_SIZE, TILE_SIZE, 1);
	float delta_x = 2.0 / ((float)M);	// x-resolution of grid in screen space (uv)
	float delta_y = 2.0 / ((float)N);	// y-resolution of grid in screen space (uv)

	//size_t start;
	//checkCudaErrors(cudaGraphicsMapResources(1, &vertexVBO.cudaResource, NULL));	// Map the vertex vbo as cudaGraphicsResource
	//checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &start,	// and then call the kernel to do work on it.
	//	vertexVBO.cudaResource));

	create_Grid_points_only_kernel << < BLK, THD >> >(dDptr, dEz_float, M, N, delta_x, delta_y);		// kernel call

	//checkCudaErrors(cudaGraphicsUnmapResources(1, &vertexVBO.cudaResource, NULL));	// Unmap the vertex vbo as cudaGraphicsResource
}																					// so that OpenGL can have it back. have it back.
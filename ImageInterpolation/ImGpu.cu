/**********************************************************************************/
/* The MIT License(MIT)                                                           */
/*                                                                                */
/* Copyright(c) 2016-2016 Matthieu Vendeville                                     */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files(the "Software"), to deal   */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and / or sell    */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions :                      */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE     */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#include "ImGpu.h"
#include <stdio.h>
#include <iostream>

__global__ void ComputeXDest(float *xdest, float WidthScaleFactor)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	xdest[idx] = (float)(idx + .5)*WidthScaleFactor;
}

__global__ void ComputeYDest(float *ydest, float HeightScaleFactor)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	ydest[idx] = (float)(idx + .5)*HeightScaleFactor;
}

__global__ void KernelInterpolateNN(void *pxl, void *new_pxl, float *xdest, float *ydest, unsigned short new_width, unsigned short width)
{
	unsigned short  XRounded, YRounded;

	// X and Y are pixels coordinates in destination image
	unsigned short X = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned short Y = (blockIdx.y * blockDim.y) + threadIdx.y;

	// XRounded and YRounded are coordinates of the nearest neighbor in the original image */
	XRounded = (unsigned short)xdest[X];
	YRounded = (unsigned short)ydest[Y];

	*((char*)new_pxl + X + Y*new_width) = *((char*)pxl + XRounded + YRounded*width);
}

#define ImPxl(IM,X,Y,W)     *((unsigned char*)IM + (X) + (Y)*W)

__global__ void KernelInterpolateBilinear(void *pxl, void *new_pxl, unsigned short new_width, unsigned short width, unsigned short new_height, unsigned short height, float *xd, float *yd)
{
	unsigned short  X = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned short  Y = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned short	Xp1, Xp2, Xp3, Xp4;
	unsigned short	Yp1, Yp2, Yp3, Yp4;
	unsigned short	Integer;

	/* Compute scaling factor for each dimension */
	float xdest, ydest;
	double alphax, alphay;

	/*
	* xdest and ydest are coordinates of destination pixel in the original image
	*/
	xdest = xd[X];
	ydest = yd[Y];

	/* Processing pixels in the top left corner */
	if ((xdest < 0.5) && (ydest < 0.5))
	{
		ImPxl(new_pxl, X, Y, new_width) = ImPxl(pxl, 0, 0, width);
	}

	/* Processing pixels in the top center */
	if ((xdest > 0.5) && (ydest < 0.5) && (xdest < (width - 1 + 0.5)))
	{
		/* Compute Alpha x value used to perform interpolation */

		Integer = (unsigned short)(xdest - 0.5);
		alphax = (float)((xdest - 0.5) - Integer);
		Xp1 = Integer;
		Xp2 = Xp1 + 1;

		// (1 - t)*v0 + t*v1; // fma(t, v1, fma(-t, v0, v0))

		/* Perform bilinear interpolation */
		ImPxl(new_pxl, X, Y, new_width) = (unsigned char)((1 - alphax)*ImPxl(pxl, Xp1, 0, width) + alphax*ImPxl(pxl, Xp2, 0, width));
		// ImPxl(new_pxl, X, Y, new_width) = (unsigned char)fma(alphax, ImPxl(pxl, Xp2, 0, width), fma(-alphax, ImPxl(pxl, Xp1, 0, width), ImPxl(pxl, Xp1, 0, width)));

	}

	/* Processing pixels in the top right corner */
	if ((ydest < 0.5) && (xdest >(width - 1 + 0.5)))
	{
		/* Taking last pixel of the first row */
		ImPxl(new_pxl, X, Y, new_width) = ImPxl(pxl, width - 1, 0, width);
	}

	/* Processing pixels in left side, center */
	if ((xdest < 0.5) && (ydest > 0.5) && (ydest < (height - 1 + 0.5)))
	{
		/* Compute Alpha y value used to perform interpolation */
			Integer = (unsigned short)(ydest - 0.5);
		alphay = (float)((ydest - 0.5) - Integer);

		Yp1 = Integer;
		Yp3 = Yp1 + 1;

		/* Perform bilinear interpolation */
		ImPxl(new_pxl, X, Y, new_width) = (unsigned char)((1 - alphay)*ImPxl(pxl, 0, Yp1, width) + alphay*ImPxl(pxl, 0, Yp3, width));
	}

	/* Processing pixels in the center */
	if ((xdest > 0.5) && (ydest > 0.5) && (xdest < (width - 1 + 0.5)) && (ydest < (height - 1 + 0.5)))
	{
		/*
		* Compute Alpha x and Alpha y values used to perform interpolation
		*/
		Integer = (unsigned short)(xdest - 0.5);
		alphax = (float)((xdest - 0.5) - Integer);
		Xp1 = Xp3 = Integer;
		Xp2 = Xp4 = Xp1 + 1;

		Integer = (unsigned short)(ydest - 0.5);
		alphay = (float)((ydest - 0.5) - Integer);

		Yp1 = Yp2 = Integer;
		Yp3 = Yp4 = Yp1 + 1;

		/* Perform bilinear interpolation */
		ImPxl(new_pxl, X, Y, new_width) = (unsigned char)((1 - alphax)*(1 - alphay)*ImPxl(pxl, Xp1, Yp1, width) + alphax*(1 - alphay)*ImPxl(pxl, Xp2, Yp2, width) + (1 - alphax)*alphay*ImPxl(pxl, Xp3, Yp3, width) + alphay*alphax*ImPxl(pxl, Xp4, Yp4, width));
	}

	/* Processing pixels in right side, center */
	if ((xdest > (width - 1 + 0.5)) && (ydest > 0.5) && (ydest < (height - 1 + 0.5)))
	{
		/*
		* Compute Alpha y values used to perform interpolation
		*/
		Integer = (unsigned short)(ydest - 0.5);
		alphay = (float)((ydest - 0.5) - Integer);

		Yp1 = Yp2 = Integer;
		Yp3 = Yp4 = Yp1 + 1;

		/* Perform bilinear interpolation */
		ImPxl(new_pxl, X, Y, new_width) = (unsigned char)((1 - alphay)*ImPxl(pxl, (width - 1), Yp1, width) + alphay*ImPxl(pxl, (width - 1), Yp3, width));
	}

	/* Processing pixels in the lower left corner */
	if ((xdest < 0.5) && (ydest >(height - 1 + 0.5)))
	{
		ImPxl(new_pxl, X, Y, new_width) = ImPxl(pxl, 0, height - 1, width);
	}

	/* Processing pixels in bottom , center */
	if ((xdest > 0.5) && (xdest < (width - 1 + 0.5)) && (ydest >(height - 1 + 0.5)))
	{
		/*
		* Compute Alpha x values used to perform interpolation
		*/
		Integer = (unsigned short)(xdest - 0.5);
		alphax = (float)((xdest - 0.5) - Integer);
		Xp1 = Integer;
		Xp2 = Xp1 + 1;

		/* Perform bilinear interpolation */
		ImPxl(new_pxl, X, Y, new_width) = (unsigned char)((1 - alphax)*ImPxl(pxl, Xp1, height - 1, width) + alphax*ImPxl(pxl, Xp2, height - 1, width));
	}

	/* Processing pixels in the lower right corner */
	if ((xdest > (width - 1 + 0.5)) && (ydest > (height - 1 + 0.5)))
	{
		ImPxl(new_pxl, X, Y, new_width) = ImPxl(pxl, width - 1, height - 1, width);
	}

	return;
}



void ImGpu::InterpolateNN(unsigned short new_width, unsigned short new_height)
{
	void *dev_new_pxl;
	cudaError_t cudaStatus;
	float *xdest, *ydest;

	/* Compute scaling factor for each dimension */
	float HeightScaleFactor = ((float)height / (float)new_height);
	float WidthScaleFactor = ((float)width / (float)new_width);

	// Allocate GPU buffers for the buffers of pixels on the GPU.
	cudaStatus = cudaMalloc((void**)&dev_new_pxl, new_width *new_height *dimension * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&xdest, new_width * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&ydest, new_height * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	// Using Streams and events API to synchronize kernel calls
	{
		dim3 threadsPerBlock(16, 16);  // 64 threads
		dim3 numBlocks((new_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (new_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
		cudaStream_t streamA, streamB, streamC;;
		cudaEvent_t event, event2;

		cudaStreamCreate(&streamA);
		cudaStreamCreate(&streamB);
		cudaStreamCreate(&streamC);

		cudaEventCreate(&event);
		cudaEventCreate(&event2);

		cudaEventRecord(event, streamA);
		cudaEventRecord(event2, streamB);

		ComputeXDest << < (new_width + 96 - 1) / 96, 96, 0, streamA >> > (xdest, WidthScaleFactor);
		ComputeYDest << < (new_height + 96 - 1) / 96, 96,0, streamB >> > (ydest, HeightScaleFactor);

		// Do not call KernelInterpolateNN until computation needed has been done in ComputeXDest and ComputeYDest
		// Ensuring correct stream synchro with cudaStreamWaitEvent
		cudaStreamWaitEvent(streamC, event, 0);
		cudaStreamWaitEvent(streamC, event2, 0);
		KernelInterpolateNN << < numBlocks, threadsPerBlock,0, streamC >> > (dev_pxl, dev_new_pxl, xdest, ydest, new_width, width);
	}

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

	// Free all resources
	cudaFree(dev_pxl);
	cudaFree(xdest);
	cudaFree(ydest);

	dev_pxl = dev_new_pxl;

	width = new_width;
	height = new_height;

	return;
Error:
	cudaFree(dev_new_pxl);
	cudaFree(xdest);
	cudaFree(ydest);
}

void ImGpu::InterpolateBilinear(unsigned short new_width, unsigned short new_height)
{
	void *dev_new_pxl;
	cudaError_t cudaStatus;
	float *xdest, *ydest;

	/* Compute scaling factor for each dimension */
	float HeightScaleFactor = ((float)height / (float)new_height);
	float WidthScaleFactor = ((float)width / (float)new_width);

	// Allocate GPU buffers for the buffers of pixels on the GPU.
	cudaStatus = cudaMalloc((void**)&dev_new_pxl, new_width *new_height *dimension * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&xdest, new_width * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&ydest, new_height * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	{
		dim3 threadsPerBlock(16, 16);  // 64 threads
		dim3 numBlocks((new_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (new_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
		cudaStream_t streamA, streamB, streamC;;
		cudaEvent_t event, event2;

		cudaStreamCreate(&streamA);
		cudaStreamCreate(&streamB);
		cudaStreamCreate(&streamC);

		cudaEventCreate(&event);
		cudaEventCreate(&event2);

		ComputeXDest << < (new_width + 96 - 1) / 96, 96, 0, streamA >> > (xdest, WidthScaleFactor);
		cudaEventRecord(event, streamA);

		ComputeYDest << < (new_height + 96 - 1) / 96, 96, 0, streamB >> > (ydest, HeightScaleFactor);
		cudaEventRecord(event2, streamB);

		// Do not call KernelInterpolateNN until computation needed has been done in ComputeXDest and ComputeYDest
		// Ensuring correct stream synchro with cudaStreamWaitEvent
		cudaStreamWaitEvent(streamC, event, 0);
		cudaStreamWaitEvent(streamC, event2, 0);

		KernelInterpolateBilinear << < numBlocks, threadsPerBlock, 0, streamC >> > (dev_pxl, dev_new_pxl, new_width, width, new_height, height, xdest, ydest);
	}

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

	// Free all resources
	cudaFree(dev_pxl);
	cudaFree(xdest);
	cudaFree(ydest);

	dev_pxl = dev_new_pxl;

	width = new_width;
	height = new_height;

	return;
Error:
	cudaFree(dev_new_pxl);
	cudaFree(xdest);
	cudaFree(ydest);
}
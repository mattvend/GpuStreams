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

ImGpu::ImGpu(unsigned short width, unsigned short height, unsigned short bpp, unsigned short dimension)
{
	width = width;
	height = height;
	bpp = bpp;
	dimension = 1;
	cudaError_t cudaStatus;

	/* Allocate memory for the pixels on the Gpu */
	if (8 == bpp)
	{
		cudaStatus = cudaMalloc((void**)&dev_pxl, width *height *dimension * sizeof(char));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		cudaMemset(dev_pxl, 255, sizeof(char) * width *height *dimension);
	}
	else if (16 == bpp)
	{
		cudaStatus = cudaMalloc((void**)&dev_pxl, width *height *dimension * sizeof(unsigned short));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		cudaMemset(dev_pxl, 255, sizeof(unsigned short) * width *height *dimension);
	}

Error:
	cudaFree(dev_pxl);
}

ImGpu::ImGpu(const ImGpu &obj)
{
	width = obj.width;
	height = obj.height;
	bpp = obj.bpp;
	dimension = 1;

	cudaError_t cudaStatus;

	/* Allocate memory for the pixels on the Gpu */
	if (8 == bpp)
	{
		cudaStatus = cudaMalloc((void**)&dev_pxl, width *height *dimension * sizeof(char));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_pxl, obj.dev_pxl, width *height *dimension * sizeof(char), cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}
	else if (16 == bpp)
	{
		cudaStatus = cudaMalloc((void**)&dev_pxl, width *height *dimension * sizeof(unsigned short));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_pxl, obj.dev_pxl, width *height *dimension * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}
	return;
Error:
	cudaFree(dev_pxl);
}


ImGpu::ImGpu(const char* filename)
{
	FILE *fp = 0;
	int t1, t2, t3, t4;
	cudaError_t cudaStatus;

	sscanf_s(filename, "%dx%dx%dx%d_", &t1, &t2, &t3, &t4);

	width = t1;
	height = t2;
	bpp = t3;
	dimension = t4;

	void *pxl = 0;

	/* Allocate memory for the pixels on the Gpu */
	if (8 == bpp)
	{
		cudaStatus = cudaMalloc((void**)&dev_pxl, width *height *dimension * sizeof(char));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		cudaMemset(dev_pxl, 255, sizeof(char) * width *height *dimension);
		pxl = new char[sizeof(char) * width *height *dimension];
	}
	else if (16 == bpp)
	{
		cudaStatus = cudaMalloc((void**)&dev_pxl, width *height *dimension * sizeof(unsigned short));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		cudaMemset(dev_pxl, 255, sizeof(unsigned short) * width *height *dimension);
		pxl = new unsigned short[sizeof(unsigned short) * width *height *dimension];
	}

	/*
	* Open the file to read the pixels
	*/
	fopen_s(&fp, filename, "rb"); /* open for reading */

	if (0 != fp){
		std::fread(pxl, sizeof(unsigned char), width*height*dimension, fp);
		fclose(fp); /* close the file */
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_pxl, pxl, width *height *dimension * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	delete(pxl);
	return;
Error:
	cudaFree(dev_pxl);
	//delete(pxl);
}

ImGpu::~ImGpu(void)
{
	cudaFree(dev_pxl);
}


void ImGpu::Save2RawFile(const char* filename)
{
	FILE *fp;
	void *pxl = 0;
	cudaError_t cudaStatus;

	/* Allocate memory for temporary buffer on CPU */
	if (8 == bpp)
	{
		pxl = new char[sizeof(char) * width *height *dimension];
	}
	else if (16 == bpp)
	{
		pxl = new unsigned short[sizeof(unsigned short) * width *height *dimension];
	}

	sprintf_s(raw_file_name, "%dx%dx%dx%d_%s", width, height, bpp, dimension, filename);

	fopen_s(&fp, raw_file_name, "wb"); /* open for writing */

	if (8 == bpp)
	{
		cudaStatus = cudaMemcpy(pxl, dev_pxl, width *height *dimension * sizeof(char), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		fwrite(pxl, sizeof(char), width *height *dimension, fp);
		
	}
	else if (16 == bpp)
	{
		cudaStatus = cudaMemcpy(pxl, dev_pxl, width *height *dimension * sizeof(unsigned short), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		fwrite(pxl, sizeof(unsigned short), width *height *dimension, fp);
	}

	fclose(fp); /* close the file before ending program */

Error:
	delete(pxl);

	return;
}

void ImGpu::PrintRawFileName()
{
	std::cout << raw_file_name << '\n';
}
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
#include <cstdio>

ImGpu::ImGpu(const ImGpu &obj)
{
    width = obj.width;
    height = obj.height;
    bpp = obj.bpp;
    dimension = 1;

    cudaError_t cudaStatus;
    int mySizeOf;

    (8 == bpp) ? mySizeOf = sizeof(char) : mySizeOf = sizeof(unsigned short);

#if USE_STREAMS
    pStream = (cudaStream_t *) malloc(1 * sizeof(cudaStream_t));
    cudaStreamCreate(pStream);
#endif

    /* Allocate memory for the pixels on the Gpu */

    cudaStatus = cudaMalloc((void**)&dev_pxl, width *height *dimension * mySizeOf);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
#if USE_STREAMS
    cudaStatus = cudaMemcpyAsync(dev_pxl, obj.dev_pxl, width *height *dimension * mySizeOf, cudaMemcpyDeviceToDevice, *pStream);
#else
   cudaStatus = cudaMemcpy(dev_pxl, obj.dev_pxl, width *height *dimension * mySizeOf, cudaMemcpyDeviceToDevice);
#endif
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    return;
Error:
    cudaFree(dev_pxl);
}


ImGpu::ImGpu(std::string filename)
{
    FILE *fp = 0;
    int t1, t2, t3, t4;
    cudaError_t cudaStatus;
    int mySizeOf;

    (8 == bpp) ? mySizeOf = sizeof(char) : mySizeOf = sizeof(unsigned short);

    sscanf(filename.c_str(), "%dx%dx%dx%d_", &t1, &t2, &t3, &t4);

    width = t1;
    height = t2;
    bpp = t3;
    dimension = t4;

    void *pxl = 0;

#if USE_STREAMS
    pStream = (cudaStream_t *) malloc(1 * sizeof(cudaStream_t));
    cudaStreamCreate(pStream);
#endif

    /* Allocate memory for the pixels on the Gpu */
    cudaStatus = cudaMalloc((void**)&dev_pxl, width *height *dimension * mySizeOf);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaMallocHost(&pxl, mySizeOf * width *height *dimension);


    /*
    * Open the file to read the pixels
    */
    fp = fopen(filename.c_str(), "rb"); /* open for reading */

    if (0 != fp){
        std::fread(pxl, mySizeOf, width*height*dimension, fp);
        fclose(fp); /* close the file */
    }

    // Copy input vectors from host memory to GPU buffers.
    #if USE_STREAMS
        cudaStatus = cudaMemcpyAsync(dev_pxl, pxl, width *height *dimension * mySizeOf, cudaMemcpyHostToDevice, *pStream);
    #else
        cudaStatus = cudaMemcpy(dev_pxl, pxl, width *height *dimension * mySizeOf, cudaMemcpyHostToDevice);
    #endif

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaFreeHost(pxl);
    return;
Error:
    cudaFree(dev_pxl);
}

ImGpu::~ImGpu(void)
{
    cudaFree(dev_pxl);
#if USE_STREAMS
    cudaStreamDestroy(*pStream);
#endif
}


void ImGpu::Save2RawFile(std::string filename)
{
    FILE *fp;
    void *pxl = 0;
    cudaError_t cudaStatus;
    int mySizeOf;

    (8 == bpp) ? mySizeOf = sizeof(char) : mySizeOf = sizeof(unsigned short);

    /* Allocate memory for temporary buffer on CPU */
    cudaMallocHost(&pxl, mySizeOf * width *height *dimension);


    sprintf(raw_file_name, "%dx%dx%dx%d_%s", width, height, bpp, dimension, filename.c_str());

    fp = fopen(raw_file_name, "wb"); /* open for writing */

#if USE_STREAMS
    cudaStatus = cudaMemcpyAsync(pxl, dev_pxl, width *height *dimension * mySizeOf, cudaMemcpyDeviceToHost,*pStream);
    cudaStreamSynchronize(*pStream);
#else
    cudaStatus = cudaMemcpy(pxl, dev_pxl, width *height *dimension * mySizeOf, cudaMemcpyDeviceToHost);
#endif

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    fwrite(pxl, mySizeOf, width *height *dimension, fp);


    fclose(fp); /* close the file before ending program */

Error:
    cudaFreeHost(pxl);

    return;
}

void ImGpu::PrintRawFileName()
{
    std::cout << raw_file_name << '\n';
}

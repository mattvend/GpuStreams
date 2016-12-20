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

#include <stdio.h>
#include "ImCpu.h"
#include "ImGpu.h"
#include "cuda_profiler_api.h"
#include <time.h>
#include <iostream>
using namespace std;

int main(int argc, char** argv) 
{
	Im *Im1,*Im2;
	cudaError_t cudaStatus;
	clock_t begin_time, end_time = 0;
	int i;
	
	char *device_type = "gpu";
	char *interpolation_type = "nn";
	char *in_file = "512x512x8x1_lena.dat";
	char *out_file = "lena_out.dat";

	int iterations = 10;
	int new_width = 8000;
	int new_height = 4000;
	
	if (argc > 1 ){
		device_type = argv[1];
		iterations = atoi(argv[2]);
		interpolation_type = argv[3];
		in_file  = argv[4];
		out_file = argv[5];
		new_width = atoi(argv[6]);
		new_height = atoi(argv[7]);

		std::cout << "Using device: " << device_type <<'\n';
		std::cout << "Nb iterations: " << iterations << '\n';
		std::cout << "Interplation types: " << interpolation_type << '\n';
		std::cout << "Input file: " << in_file << '\n';
		std::cout << "Output file: " << out_file << '\n';
		std::cout << "New width: " << new_width << '\n';
		std::cout << "New height: " << new_height << '\n';
	}

	//
	// Initialise GPU
	//
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//
	// Init instance, depending on process happening on GPU or CPU
	//
	if (strcmp(device_type, "cpu") == 0){
		Im1 = new ImCpu(in_file);
		std::cout << "Creating Imcpu instance" << '\n';
	}else{
		Im1 = new ImGpu(in_file);
		std::cout << "Creating Imgpu instance" << '\n';
	}

	//
	// Perform and profile interpolation x times 
	//

	for (i = 0; i < iterations; i++){
		Im2 = Im1->clone();
		if (strcmp(interpolation_type, "nn") == 0)
		{
			begin_time = clock();
			Im2->InterpolateNN(new_width, new_height);
			end_time += clock() - begin_time;
		}
		else
		{
			begin_time = clock();
			Im2->InterpolateBilinear(new_width, new_height);
			end_time += clock() - begin_time;
		}
		delete(Im2);
	}


	std::cout << float(end_time) / CLOCKS_PER_SEC << '\n';

	//
	// Save processed imaged
	//
	if (strcmp(interpolation_type, "nn") == 0)
	{
		Im1->InterpolateNN(new_width, new_height);
	}
	else
	{
		Im1->InterpolateBilinear(new_width, new_height);
	}

	Im1->Save2RawFile(out_file);
	Im1->PrintRawFileName();

	exit(0);
	
}


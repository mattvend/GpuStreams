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


#include "ImCpu.h"
#include "ImGpu.h"
#include "cuda_profiler_api.h"
#include <time.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <sstream>
#include <thread>

using namespace std;

struct arg_struct {
    string device_type;
    string interpolation_type;
    string in_file;
    string out_file;
    int iterations;
    int new_width;
    int new_height;
};


void *Resize( void *args)
{
    Im *Im1,*Im2;
    cudaError_t cudaStatus;
    clock_t begin_time, end_time = 0;
    int i;
    struct arg_struct *parameters = (struct arg_struct *)args;

    cout << "Using device: " << parameters->device_type <<'\n';
    cout << "Nb iterations: " << parameters->iterations << '\n';
    cout << "Interplation types: " << parameters->interpolation_type << '\n';
    cout << "Input file: " << parameters->in_file << '\n';
    cout << "Output file: " << parameters->out_file << '\n';
    cout << "New width: " << parameters->new_width << '\n';
    cout << "New height: " << parameters->new_height << '\n';

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
    if ( parameters->device_type == "cpu" ){
        Im1 = new ImCpu(parameters->in_file.c_str());
        cout << "Creating Imcpu instance" << '\n';
    }else{
        Im1 = new ImGpu(parameters->in_file.c_str());
        cout << "Creating Imgpu instance" << '\n';
    }
    
    
    // Perform and profile interpolation parameters->iterations times 
    for (i = 0; i < parameters->iterations; i++){
        Im2 = Im1->clone();
        if ( parameters->interpolation_type == "nn")
        {
            begin_time = clock();
            Im2->InterpolateNN(parameters->new_width, parameters->new_height);
            end_time += clock() - begin_time;
        }
        else
        {
            begin_time = clock();
            Im2->InterpolateBilinear(parameters->new_width, parameters->new_height);
            end_time += clock() - begin_time;
        }
        delete(Im2);
    }

    if ( parameters->iterations != 0)
    {
        cout << float(end_time) / CLOCKS_PER_SEC << '\n';
    }else
    {
        cout << 0 << '\n';
    }

    //
    // Save processed imaged
    //
    if ( parameters->interpolation_type == "nn")
    {
        Im1->InterpolateNN(parameters->new_width, parameters->new_height);
    }
    else
    {
        Im1->InterpolateBilinear(parameters->new_width, parameters->new_height);
    }

    Im1->Save2RawFile(parameters->out_file.c_str());
    Im1->PrintRawFileName();
    delete(Im1);

}



int main(int argc, char** argv) 
{
    cudaError_t cudaStatus;
 
    int i;
    int NbFiles = 10;
    std::thread threads[NbFiles];
    struct arg_struct ThreadArguments[NbFiles];

    /*
     *   Let's start 
     */

    // reset Gpu
    cudaDeviceReset();


    // For each thread, create an argument struct
    for (i=0; i<NbFiles; i++)
    {
        std::string filename_out = "lenaout" + std::to_string(i) + ".dat";

        if (argc > 1 ){
            NbFiles = 1;
            ThreadArguments[i].device_type = std::string(argv[1]);
            ThreadArguments[i].iterations  = atoi(argv[2]);
            ThreadArguments[i].interpolation_type = atoi(argv[3]);
            ThreadArguments[i].in_file = std::string(argv[4]);
            ThreadArguments[i].out_file = std::string(argv[5]);
            ThreadArguments[i].new_width = atoi(argv[6]);
            ThreadArguments[i].new_height = atoi(argv[7]);
        }else{
            ThreadArguments[i].device_type = std::string({"gpu"});
            ThreadArguments[i].interpolation_type = std::string({"nn"});
            ThreadArguments[i].in_file = std::string({"512x512x8x1_Lena.dat"});

            ThreadArguments[i].out_file = filename_out;
            ThreadArguments[i].iterations = 0;
            ThreadArguments[i].new_width = 256;
            ThreadArguments[i].new_height = 256;
        }
    }


    /*
     *   Create NbFiles threads to do parallel processing
     */

    for (i=0; i<NbFiles; i++)
    {
        threads[i] = std::thread(Resize, &ThreadArguments[i]);
    }

    
    /*
     *   Wait for all threads to terminate properly 
     */
    for (i=0; i<NbFiles; i++)
    {
        threads[i].join();
    }
    
    /*
     *   Free all allocated resources and exit properly
     */
    for (i=0; i<NbFiles; i++)
    {
        ThreadArguments[i].device_type.erase();
        ThreadArguments[i].in_file.erase();        
        ThreadArguments[i].interpolation_type.erase();
        ThreadArguments[i].out_file.erase();
    }

    exit(0);    
}


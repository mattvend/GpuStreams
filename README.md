# GpuStreams

This is work in progress 

### Goal of this project

The previous ![Gpu][Gpu] project was ok, but there was still issues I wanted to tackle. Particularly:
- there were issues with certain parameters, those issues are solved
- it is now ported under Linux
- the code was not as clean and commented as it could have been
- I  felt somehow with the previous ![project][Gpu] that the usage of ![streams][CUDA streams] was too artificial. I hope to propose a better example with this rework.

Finally, the main goal of the project is still to benchmark GPU againt CPU interpolation algorithms using Lena as input image. For the moment, benchmarking is only done with the Nearest Neighbor and Bilinear interpolations.

### Intended audience
The project has been ported to Ubuntu 16.04 for people and teams trying to get rid of Windows as their developemnt environment. :)

### My Setup
hardware
- CPU: PC i7 Intel core @ 3.60 GHz,  64-bits Operating System, 16 GB memory 
- GPU: NVIDIA [Quadro 600], which is a really old card
```
Detected 1 CUDA Capable device(s)

Device 0: "Quadro 600"
  CUDA Driver Version / Runtime Version          8.0 / 8.0
  CUDA Capability Major/Minor version number:    2.1
  Total amount of global memory:                 957 MBytes (1003225088 bytes)
  ( 2) Multiprocessors, ( 48) CUDA Cores/MP:     96 CUDA Cores
  GPU Max Clock rate:                            1280 MHz (1.28 GHz)
  Memory Clock rate:                             800 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 131072 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65535), 3D=(2048, 2048, 2048)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 32768
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (65535, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
```
Software
- Ubuntu 16.04 LTS, CUDA SDK 8.0 installed
- Python 3.5.2 

### What has been changed since Gpu
- the project has been ported on Ubuntu 16.04, all references to windows projects have been removed
- use c++ strings (not everywhere)
- use pinned memory to speed up memory transfers between the Host and the Device (no more malloc()/free())

### Directory Structure
- /ImageInterpolation contains the source code for this example
- /Release contains the benchmark scripts, the checked in executables needed for the experiment, and the results

### Architecture and design considerations

#### Building
```shell
cd ../ImageInterpolation/; make clean; make
```
#### Design considerations
A simple test applications receiving arguments performs interpolation whose parameters depends on received arguments.
The test application returns to the user an elapsed duration in seconds and a file name

A simple Python script exercise the test app and hence can benchamrk one solution against the other

#### C++
The Im abstract class represents our image class with two interpolation methods: NN and bilinear  
ImCpu and ImGpu implement the interface, with code running respectively on the CPU and on the GPU

Despite some code already there for other image format, note that the classes only works for 8 bpp / 1 channel images, ie grey images.

The test application uses ImCpu or ImGpu depending on the received arguments. All remaining code stays the same as both classes implement the same abstract interface

#### Timing
I deliberately choosed to exclude the memory transfers between the Gpu and the Host when counting the time. On a more complex processing chain, pixels do not travel constantly from the Gpu to the Host between 2 operations. They are only retrieved when needed on the host. This is why I do not count memory transfers between the Host and the Gpu.

## Benchmarking cpu vs gpu

Once test application is built, simply execute the python script with the following command: 
```python
python3 Benchmark.py
```

### Results
Here are the results, produced by Benchmark.py
![Results][Results]

### Comments
- The results above are for 20 iterations. To get the time needed to do interpolation only once, divide by 20.
- The Gpu version of the NN interpolation is only 5 times faster when the Bilinear interpolation runs 6 times faster.
- I only did the tests for the Lena image (512*512), interpolated to a (8000,4000) image. Different interpolation parameters should provide a better overall picture
- Both interpolation algorithms are done in a naive way
- Cpu vs Gpu benchamrking seems to be tricky as the figures obtained depends obviously on the setup. In my case, as the GPU used is an old one (5 years older than the CPU, huge difference in the tech world), it makes sense to have a CPU that can compete against a GPU. The ![Quadro 600][Quadro 600] card has only 96 cores, and is definitely not a fast card, see this review: ![Quadro 600 review][Quadro 600 review]
- CUDA code can be improved using intrinsics  


## Using Streams

Using ![CUDA streams][CUDA streams] and ![CUDA events][CUDA events]

### how to use CUDA streams

### Results
#### Without Streams
![NonStreams][NonStreams]
#### With Streams
![Streams][Streams]
### Comments
There are many possible reasons to not get the expexted results
- 
- 

### simpleStreams
![simpleStreams][simpleStreams]

## Conclusion

[Lena]: http://www.cosy.sbg.ac.at/~pmeerw/Watermarking/lena_color.gif "Lena"

[Results]: /Release/CpuVsGpu.png
[Streams]: /Release/Streams.png
[NonStreams]: /Release/NonStream.png
[simpleStreams]: /Release/simpleStreams.png

[xkcd]: http://xkcd.com/
[Anaconda]: https://www.continuum.io/why-anaconda
[Quadro 600]: http://www.nvidia.com/object/product-quadro-600-us.html
[Quadro 600 review]: https://www.dpreview.com/forums/post/53081447
[Benchmark.py]: /Release/Benchmark.py
[Matplotlib]: http://matplotlib.org/
[CUDA streams]: http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM
[CUDA events]: http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT
[Gpu]: https://github.com/mattvend/Gpu

# GpuStreams

This is work in progress 

### Goal of this project
If I was satisfied with the previous ![Gpu][Gpu] project, there was still some issues I wanted to tackle. Particularly:
- it is now ported under Linux Ubuntu 16.04, all references to the Windows project have been removed
- there were issues with certain parameters, those issues are solved
- the code was not as clean and commented as it could have been
- I  felt somehow with the previous ![project][Gpu] that the usage of ![streams][CUDA streams] was too artificial. I hope to propose a better example with this rework.

Finally, the main goal of the project is still to benchmark GPU againt CPU interpolation algorithms using Lena as input image. For the moment, benchmarking is only done with the Nearest Neighbor and Bilinear interpolations.

### Intended audience
The project has been ported to Ubuntu 16.04 for people and teams trying to get rid of Windows as their developement environment. :)

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

### Directory Structure
- ImageInterpolation contains the source code for this benchamrking project
- Release contains the benchmark scripts, the checked-in executables needed for the experiment, and result screenshots

#### Building
To regenerate all the executables:
```shell
cd ../ImageInterpolation/; make clean; make
```
## Benchmarking cpu vs gpu
Interpolate, in the release directory, is the executable used for this benchmarking, it is built from Main.cpp.

#### Design considerations
A simple test application, interpolate.exe, receiving arguments performs interpolation whose parameters depends on received arguments.
The test application returns to the user an elapsed duration in seconds and a file name

A simple Python script exercise the test app and hence can benchamrk one solution against the other

#### C++
The Im abstract class represents our image class with two interpolation methods: NN and bilinear  
ImCpu and ImGpu implement the interface, with code running respectively on the CPU and on the GPU

Despite some code already there for other image format, note that the classes only works for 8 bpp / 1 channel images, ie grey images.

The test application uses ImCpu or ImGpu depending on the received arguments. All remaining code stays the same as both classes implement the same abstract interface

#### Timing
I deliberately choosed to exclude the memory transfers between the Gpu and the Host when counting the time for the following reason: on a more complex processing chain, pixels do not travel constantly from the Gpu to the Host between 2 operations. They are only retrieved when needed on the host.

Once test application is built, simply execute the python script with the following command: 
```python
python3 Benchmark.py
```

### Results
Here are the results, produced by Benchmark.py
![Results][Results]

### Comments
- The results above are for 20 iterations. To get the time needed to do interpolation only once, divide by 20.
- The Gpu version of the NN interpolation is 5 times faster when the Bilinear interpolation runs 6 times faster.
- I only did the tests for the Lena image (512*512), interpolated to a (8000,4000) image. Different interpolation parameters should provide a better overall picture
- Both interpolation algorithms are done in a naive way
- Cpu vs Gpu benchamrking seems to be tricky as the figures obtained depends obviously on the setup. In my case, as the GPU used is an old one (5 years older than the CPU, huge difference in the tech world), it makes sense to have a CPU that can compete against a GPU. The ![Quadro 600][Quadro 600] card has only 96 cores, and is definitely not a fast card, see this review: ![Quadro 600 review][Quadro 600 review]
- CUDA code can be improved using intrinsics  

## Using Streams
The second part of the experiment consists in using ![CUDA streams][CUDA streams] in the hope of improving performances. I will therefore compare two GPU applications, both running on the device, with and without Streams.

### What is a stream 
A stream is a queue of device work. It is possible to take advantage of ![CUDA streams][CUDA streams] and ![CUDA events][CUDA events] in 2 different ways:
- Execute concurent kernels on the device, thus enabling parallel processing. To do so, kernel calls have to be placed in different non default streams. Kernel calls in the same stream are automatically synchronous, while using events allow to synchronize kernel calls in two separate streams.
- and/or execute concurrent memory copies from/to the GPU.

As I only have a limited number of kernel to launch (3 to be accurate) in order to perform interpolation, I will use the second method to improve performances.

### how to use CUDA streams for concurent memory copies
The code needs to be modified and follow those guidelines:
- use non-default stream for the memory copy
- use host pinned memory
- call the async method
- 1 memcopy occuring in the same direction at the same time

MainThread.cpp does many interpolation in parallel by using CPU threads on the Host. In each thread, a specific stream is allocated, in the hope of allowing parallel processing across the different streams.

To be clearer, for each interpolation, I copy memory to the device, launch the 3 kernels and copy the memory back. Because it uses the same stream for the 3 operations, I don't expect any performance improvement here, but I get rid of any synchronisation issue. The gain is expected to occur across all threads, where each interpolation operation uses its own thread. There, I expect to see overlapping across memcopy operations and some kernel concurency.

In the release directory, Stream and NonStream are the same code source compiled with the following compiler switch: -D USE_STREAMS
Both test applications are built from MainThread.cpp

### Results

To verify that everything is working as expected, I used the ![nvidia profiler][nvvp], and I checked the timelines.

#### Without Streams
![NonStreams][NonStreams]

As expected, there is only the default stream, all operations are occuring sequentially as they are processed.


#### With Streams
![Streams][Streams]

This is more interesting, and can be commented:
- there are many streams created
- processing occurs in each stream: 1 memcopy from the host to the device, the processing and the copy back to the host
- at 47.5 ms, we can see some memory transfer between the host and the device overlapping with the kernel launch (yeloow and blue box)

### Is the stream version faster ?
I benchmarked the two versions, Stream and NonStream, but found out that there is no gain in this example. Knowing how things are working is nice, but I always found that knowing why things are not working as expected is nicer.

Why is there no gain at using Streams in this case ?
- overlapping memcopy and processing allows to gain a lot of time on condition that the process duration is roughly equal to the memcopy duration. Otherwise, the gain obtained by overlapping one operation with the other becomes simply too small, as the two operations have a very different duration.
- it is depending on hardware. The Quadro 600 has only 1 copy engine, meaning that it is impossible to have simulataneously 2 concurent memcopy and 1 processing.

There is also something else noticeable on the timeline: there is a lot of time spent on the host between two calls to the GPU. It doesn't help to improve performances. 

The way I wrote the program, there is probably too much time lost during memory allocation/deallocation at least for wo reasons.
- memory management functions are always slow, therefore it should be done later, when each GPU stream hs been filled with at least 3 operations that can be done in parallel
- For some functions, there is implicit synchronisation. Using cudaFree() is an example, and even working with each thread having its won stream, there is probaly time lost here.

I used a compiler siwtch to be more didactic, when I could have used a compiler option to enable the same behavior. Passing the option –default-stream per-thread to nvcc makes sure that each host thread uses its own default stream.

### simpleStreams
![simpleStreams][simpleStreams]

This comes from the examples shipped with the SDK. The timeline shows a nice overlap of kernel calls and memcopy. The measured improvement is substantial in this case, but there are many things to say about this example and why it works so well: 
- this is a well chosen example where the computation and memory transfer durations are nearly equal, therefore leading to a nice improvement.
- there is nearly no time lost between calls to kernels and memcopy functions, the code shows that everything occurs in a small for loop. Everyting is nicely packed in this case.
- there is no thread, hence no context switch slowing down things on the host.

## Conclusion

I prefer this rework to illustrate Streans usage, but it does not provide the expected improvements. If the usage of streams seems a bit more natural, the nature of the work and the way it is organized does not allow significant improvements. To go further, I will explore the memory allocation/deallocation parts and make sure that reuse is more done, and cleanup occurs at the end.


[Lena]: http://www.cosy.sbg.ac.at/~pmeerw/Watermarking/lena_color.gif "Lena"
[nvvp]: https://developer.nvidia.com/nvidia-visual-profiler
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

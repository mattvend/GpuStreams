# Gpu

## Goal of this project
Benchmark interpolation algorithms running on GPU againt their CPU version using Lena as input image.
  
![Lena][Lena]

For the moment, benchmarking is only done with the Nearest Neighbor and Bilinear interpolations

# My Setup

hardware
--------
- CPU: PC i7 Intel core @ 3.60 GHz,  64-bits Operating System, 16 GB memory 
- GPU: NVIDIA [Quadro 600], which is a really old card

Software
--------
- Windows 7, MSVC Community edition 2013, CUDA SDK installed
- Python 3.4.3 :: ![Anaconda][Anaconda] distribution 2.3.0 (64-bit)

## Building

Open the MSVC solution and simply build the Release version. ImageInterpolation.exe should appear in the Release directory.

## Architecture and design considerations

### Design considerations
A simple test applications receiving arguments performs interpolation whose parameters depends on received arguments.
The test application returns to the user an elapsed duration in seconds and a file name

A simple Python script exercise the test app and hence can benchamrk one solution against the other

### C++
The Im abstract class represents our image class with two interpolation methods: NN and bilinear  
ImCpu and ImGpu implement the interface, with code running respectively on the CPU and on the GPU

Despite some code already there for other image format, note that the classes only works for 8 bpp / 1 channel images, ie grey images.

The test application uses ImCpu or ImGpu depending on the received arguments. All remaining code stays the same as both classes implement the same abstract interface

### Timing
I deliberately choosed to exclude the memory transfers between the Gpu and the Host when counting the time. On a more complex processing chain, pixels do not travel constantly from the Gpu to the Host between 2 operations. They are only retrieved when needed on the host. This is why I do not count memory transfers between the Host and the Gpu.

## Running the benchmark
Once test application is built, simply execute the python script with the following command: 
```python
python Benchmark.py
```

## Results
Here are the results, produced by Benchmark.py
![Results][Results]

## Comments
- The results above are for 20 iterations. To get the time needed to do interpolation only once, divide by 20.
- The Gpu version of the NN interpolation is only 2 times faster when the Bilinear interpolation runs 3 times faster. It is a bit disapointing as I was expecting better performances on the GPU for both interpolations
- I only did the tests for the Lena image (512*512), interpolated to a (8000,4000) image. Different interpolation parameters should provide a better overall picture
- Both interpolation algorithms are done in a naive way
- For some parameters, there are still issues with GPU interpolation
- Cpu vs Gpu benchamrking seems to be tricky as the figures obtained depends obviously on the setup. In my case, as the GPU used is an old one (5 years older than the CPU, huge difference in the tech world), it makes sense to have a CPU that can compete against a GPU. The ![Quadro 600][Quadro 600] card has only 96 cores, and is definitely not a fast card, see this review: ![Quadro 600 review][Quadro 600 review]
- CUDA code can be improved using intrinsics  
   
#### Edit 16/8
Rewrote the NN algo using ![CUDA streams][CUDA streams] and ![CUDA events][CUDA events] and got a nice improvement. Now NN interpolation on Gpu performs 3 times faster than its Cpu conterpart. Note that all the synchro mechanism could have been done in another way by using the __syncthreads() method and some shared memory between threads.

#### Edit 19/8
Rewrote the BL algo with similar optimisations as for the NN interpolation and could divide the running time by 2 on the GPU on top on what has been already achieved. Issues still pending for some interpolations parameters, algo works well for 8000, 4000 use case. Still to be committed and debugged.

<!---
## Bonus
I found out the ![xkcd][xkcd] style while playing with ![Matplotlib][Matplotlib] ... I could not help myself to use it. Don't forget to checkout [Benchmark.py] to see how it works.
-->

[Lena]: http://www.cosy.sbg.ac.at/~pmeerw/Watermarking/lena_color.gif "Lena"
[Results]: /Release/CpuVsGpu.png
[xkcd]: http://xkcd.com/
[Anaconda]: https://www.continuum.io/why-anaconda
[Quadro 600]: http://www.nvidia.com/object/product-quadro-600-us.html
[Quadro 600 review]: https://www.dpreview.com/forums/post/53081447
[Benchmark.py]: /Release/Benchmark.py
[Matplotlib]: http://matplotlib.org/
[CUDA streams]: http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM
[CUDA events]: http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT

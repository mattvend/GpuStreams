# GpuStreans

## This is work in progress 

## Goal of this project
The previous Gpu project was ok, but there was still issues I wanted to tackle. Particularly:
- there were issues with certain parameters, those issues are solved
- it is now ported under Linux
- the code was not as clean and commented as it could have been
- the use of Streams was not the best usage, I  felt somehow that it was too artificial. With this rework, I expect to propose a better example

Finally, the goal of the project is still to benchmark GPU againt CPU interpolation algorithms using Lena as input image. I also expect to take advantage of Streams in a more natural way.

## Intended audience
The project has been ported to Ubuntu 16.04 for people and teams trying to get rid of Windows as their developemnt environment. :)

## What has been changed 
- the project has been ported on Ubuntu 16.04, all references to windows have been removed
- use c++ streams
- use pinned memory to speed up memory transfers between the Host and the Device (no more malloc()/free())

  
![Lena][Lena]

For the moment, benchmarking is only done with the Nearest Neighbor and Bilinear interpolations

# My Setup

hardware
--------
- CPU: PC i7 Intel core @ 3.60 GHz,  64-bits Operating System, 16 GB memory 
- GPU: NVIDIA [Quadro 600], which is a really old card

Software
--------
- Ubuntu 16.04 LTS, CUDA SDK 8.0 installed
- Python 3.5.2 

## Building

In Image interpolation directory, make clean; make to build the desired executables.

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

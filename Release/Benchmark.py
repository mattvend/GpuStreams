from PIL import Image
import os
import filecmp
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt


def convert_to_raw(file):
    """ Convert file to raw file.

        Args:
            file: file to convert.

        Returns:
            name of the raw file on filesystem
        """

    img = Image.open(file)
    img = img.convert('L')  # convert to 8 bits per pixels
    (x, y) = img.size

    pixels = bytearray(list(img.getdata()))

    filename, file_extension = os.path.splitext(file)
    file2 = file.replace(file_extension, '.dat')
    file_name = str(x) + 'x' + str(y) + 'x8x1' + '_' + file2

    # print(file_name)

    with open(file_name, 'wb') as f:
        f.write(pixels)

    return file_name


def convert_to_jpg(raw_file):
    """ Convert a raw file to jpg file.

        Args:
            raw_file: file to convert.

        Returns: null

        """
    match = re.match('(\d+)x(\d+)x(\d+)x(\d+)_(\w+)', raw_file)

    if match:
        print(match.group(1))
        print(match.group(2))
        print(match.group(3))
        print(match.group(4))
        print(match.group(5))
        x = int(match.group(1))
        y = int(match.group(2))
        bpp = int(match.group(3))
        dimension = int(match.group(4))
        filename = match.group(0)

    rawData = open(raw_file, 'rb').read()
    imgSize = (x, y)
    # Use the PIL raw decoder to read the data.
    # the 'F;16' informs the raw decoder that we are reading
    # a little endian, unsigned integer 16 bit data.
    # img = Image.fromstring('L', imgSize, rawData, 'raw', 'F;32')

    img = Image.frombuffer('L', imgSize, rawData, 'raw')
    img = img.rotate(180)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save(filename + ".jpg")


def interpolate(file_in, file_out, device, iterations, interpolation_type, new_width, new_height):
    """ Wrapper function on top of the interpolation executable.
        It is also a benchmarking function, it returns the name of the output
        image and the time needed to do all the iterations

        Args:
            file_in: input raw image used for the tests.
            file_out: output raw image
            device: 'cpu' or 'gpu'
            iterations: number of times we do the processing
                (we can do it iterations times, but only return 1 output image)
            interpolation_type: 'nn' or 'bl'
            new_width: output image width
            new_height: output image height

        Returns:
            a tuple containing the output image name and the time in sec needed
            to do the processing iterations times

        """

    command_string = 'ImageInterpolation.exe ' + device + ' ' + str(iterations) + ' ' + interpolation_type + ' ' + file_in + ' ' + file_out + ' ' + str(new_width) + ' ' + str(new_height)

    program_out = str(subprocess.check_output(command_string.split(), stderr=subprocess.STDOUT), 'utf-8')
    print(program_out)  # can be commented, avoid output polution
    program_out = program_out.splitlines()
    # Attention, time and file name respectively at lines 8 and 9 of the output
    seconds = float(program_out[8])
    out_file = program_out[9]

    return (seconds, out_file)


def benchmark_cpu_vs_gpu(input_raw_file):
    """ Benchmark cpu vs gpu time wise.

        Args:
            input_raw_file: input raw image used for the tests.

        Returns:
            2 tuples containing times needed to do processing on gpu and cpu
        """

    nb_iterations = 20

    (cpu1, f1) = interpolate(input_raw_file, 'cpu_nn_lena.dat', 'cpu', nb_iterations, 'nn', 8000, 4000)
    (gpu1, f2) = interpolate(input_raw_file, 'gpu_nn_lena.dat', 'gpu', nb_iterations, 'nn', 8000, 4000)
    (cpu2, f3) = interpolate(input_raw_file, 'cpu_bl_lena.dat', 'cpu', nb_iterations, 'bl', 8000, 4000)
    (gpu2, f4) = interpolate(input_raw_file, 'gpu_bl_lena.dat', 'gpu', nb_iterations, 'bl', 8000, 4000)

    # return ((cpu1/nb_iterations, cpu2/nb_iterations), (gpu1/nb_iterations, gpu2/nb_iterations))
    return ((cpu1, cpu2), (gpu1, gpu2))

def plot_graph(durations, figure_name):
    """ Plot durations in a graph

        Args:
            durations: processing durations.

        Returns:
            a file on file system
        """

    # with plt.xkcd():
    N = 2
    # cpuMeans = (1.218, 10.303)
    cpuMeans = durations[0]

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, cpuMeans, width, color='r')

    # gpuMeans = (0.669, 3.46)
    gpuMeans = durations[1]

    rects2 = ax.bar(ind + width, gpuMeans, width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Time in sec')
    ax.set_title('Duration by interpolation type and device type')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('Nearest Neighbor', 'Bilinear'))

    ax.legend((rects1[0], rects2[0]), ('Cpu', 'Gpu'))

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%.3f' % height,
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    # plt.show()
    plt.savefig(figure_name)


def check_bit_exactness(input_raw_file):
    """ Check bit exactness on interpolation executable between Gpu vs Cpu with various parameters.

        Args:
            param1: input raw image used for the tests.

        Returns:
            Prints to the output if results are bit exact

        """
    (t1, f1) = interpolate(input_raw_file, 'cpu_nn_lena.dat', 'cpu', 1, 'nn', 8000, 4000)
    (t2, f2) = interpolate(input_raw_file, 'gpu_nn_lena.dat', 'gpu', 1, 'nn', 8000, 4000)
    (t3, f3) = interpolate(input_raw_file, 'cpu_bl_lena.dat', 'cpu', 1, 'bl', 8000, 4000)
    (t4, f4) = interpolate(input_raw_file, 'gpu_bl_lena.dat', 'gpu', 1, 'bl', 8000, 4000)

    if filecmp.cmp(f1, f2, shallow=True):
        print("NN interpolation on GPU is bit exact with CPU")
    if filecmp.cmp(f3, f4, shallow=True):
        print("Bilinear interpolation on GPU is bit exact with CPU")


def exercise(input_raw_file):
    """ Exercise interpolation executable with various parameters.

        No Args:

        Returns: null
    """
    for device in ['cpu','gpu']:
        for interp in ['bl']:
            for (w,h) in ((256, 300),(2000, 1000),(1000, 2000),(8000, 4000)):
                (t, f) = interpolate(input_raw_file, device + '_' + interp + '_lena.dat', device, 1, interp, w, h)
                convert_to_jpg(f)
 

if __name__ == '__main__':

    #
    # Convert Lena Tiff image to raw format
    #
    raw_file = convert_to_raw('Lena.tiff')
    # exercise(raw_file)
    # quit()

    #
    # Check bit eaxctness between Cpu and Gpu processing
    #
    print("Checking bit-exactness between GPU processing and CPU processing")
    check_bit_exactness(raw_file)

    #
    # Perform benchmark between Cpu and Gpu processing
    # plot results in a file
    #
    print("Benchmarking execution time Cpu vs Gpu")
    durations = benchmark_cpu_vs_gpu(raw_file)
    plot_graph(durations,'CpuVsGpu.png')

    quit()

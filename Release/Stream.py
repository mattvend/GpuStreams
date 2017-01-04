from PIL import Image
import os
import filecmp
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import glob


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
    # print("examining " + raw_file)
    match = re.match('(\d+)x(\d+)x(\d+)x(\d+)_(.*)', raw_file)

    if match:
        # print(match.group(0))
        # print(match.group(2))
        # print(match.group(3))
        # print(match.group(4))
        # print(match.group(5))
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
    filename = filename.replace('.dat', '.jpg')
    img.save(filename)

if __name__ == '__main__':

    import time
    for f in glob.glob('*jpg'):
        os.remove(f)
    
    for f in glob.glob('*dat'):
        os.remove(f)
    
    raw_file = convert_to_raw('Lena.tiff')
    
    t1 = time.time()
    command_string = './MainThread '
    program_out = str(subprocess.check_output(command_string.split(), stderr=subprocess.STDOUT), 'utf-8')
    t2 = time.time()
    print(t2-t1)

    t1 = time.time()
    command_string = './MainStream '
    program_out = str(subprocess.check_output(command_string.split(), stderr=subprocess.STDOUT), 'utf-8')
    t2 = time.time()
    print(t2-t1)


    for f in glob.glob('*.dat'):
        convert_to_jpg(f)
        os.remove(f)
    
    quit()
    

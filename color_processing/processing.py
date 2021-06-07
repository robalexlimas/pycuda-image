# Libraries
import numpy as np
import os
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda import compiler
from utils import copy_host_to_device, kernel_creation


def rgb2gray(image, width, height):
    # Creating vectors for processing
    image_host = np.array(image).astype(np.float32)
    gray_host = np.zeros((height, width)).astype(np.float32)

    image_device, gray_device = copy_host_to_device(image_host, gray_host)

    path = os.path.dirname(os.path.abspath(__file__))
    parameters = {
        'width': str(width),
        'height': str(height),
        'channels': str(3)
    }
    kernel = kernel_creation(path, kernel_parameters=parameters)

    # Kernel excecution
    module = compiler.SourceModule(kernel)
    rgb2gray_function = module.get_function("rgb2gray")
    rgb2gray_function(
        gray_device,
        image_device, 
        block=(6,36, 1),
        grid = (100,8,1)
    )

    # Copy device variable to host device
    cuda.memcpy_dtoh(gray_host, gray_device)
    return gray_host

# Libraries
import numpy as np
import os
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda import compiler
from utils import copy_host_to_device, kernel_creation


def rgb2gray(image, width, height):
    # Creating vectors for processing
    image_host = np.array(image).astype(np.uint32)
    gray_host = np.zeros((height, width)).astype(np.uint32)

    image_device, gray_device = copy_host_to_device(image_host, gray_host)

    path = os.path.dirname(os.path.abspath(__file__))
    parameters = {
        'width': str(width),
        'height': str(height),
        'channels': str(3),
        'threshold': str(1)
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


def gray2bin(gray_image, threshold):
    # Definition of necessary variables 
    height_image, width_image = gray_image.shape
    rows_device = round(width_image/100)
    columns_device = round(height_image/100)

    # Creating vectors for processing
    gray_image_host = np.array(gray_image).astype(np.uint32)
    binarized_image_host = np.zeros((height_image, width_image)).astype(np.uint32)

    gray_image_device, binarized_image_device = copy_host_to_device(gray_image_host, binarized_image_host)

    path = os.path.dirname(os.path.abspath(__file__))
    parameters = {
        'height': str(height_image), 
        'width': str(width_image),
        'channels': str(3),
        'threshold': str(threshold)
    }
    kernel = kernel_creation(path, kernel_parameters = parameters)

    # Kernel excecution
    module = compiler.SourceModule(kernel)
    rgb2gray_function = module.get_function("gray2bin")
    rgb2gray_function(
        gray_image_device,
        binarized_image_device, 
        block = (rows_device, columns_device, 1),
        grid = (100, 100, 1),
    )

    # Copy device variable to host device
    cuda.memcpy_dtoh(binarized_image_host, binarized_image_device)
    return binarized_image_host
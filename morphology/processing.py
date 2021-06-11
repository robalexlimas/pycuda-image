# Libraries
import numpy as np
import os
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda import compiler
from utils import copy_host_to_device, kernel_creation


def dilatation(binarized_image, filter):
    # Definition of necessary variables 
    height_image, width_image = binarized_image.shape
    height_filter, width_filter = filter.shape
    rows_device = round(width_image / 100)
    columns_device = round(height_image / 100)

    # Creating vectors for processing
    binarized_image_host = np.array(binarized_image).astype(np.uint32)
    dilated_image_host = np.zeros((height_image, width_image)).astype(np.uint32)
    filter_host = np.array(filter).astype(np.uint32)

    binarized_image_device  = copy_host_to_device(binarized_image_host)
    dilated_image_device = copy_host_to_device(dilated_image_host)
    filter_device = copy_host_to_device(filter_host)

    path = os.path.dirname(os.path.abspath(__file__))
    parameters = {
        'height_image': str(height_image), 
        'width_image': str(width_image),
        'height_filter': str(height_filter),
        'width_filter': str(width_filter)
    }
    kernel = kernel_creation(path, kernel_parameters = parameters)

    # Kernel excecution 
    module = compiler.SourceModule(kernel)
    dilatation_function = module.get_function("dilatation")
    dilatation_function(
        binarized_image_device,
        filter_device,
        dilated_image_device, 
        block = (rows_device, columns_device, 1),
        grid = (100, 100, 1)
    ) 

    # Copy device variable to host device
    cuda.memcpy_dtoh(dilated_image_host, dilated_image_device)
    return dilated_image_host


def erosion(binarized_image, filter):
    # Definition of necessary variables 
    height_image, width_image = binarized_image.shape
    height_filter, width_filter = filter.shape
    rows_device = round(width_image / 100)
    columns_device = round(height_image / 100)

    # Creating vectors for processing
    binarized_image_host = np.array(binarized_image).astype(np.uint32)
    eroded_image_host = np.zeros((height_image, width_image)).astype(np.uint32)
    filter_host = np.array(filter).astype(np.uint32)

    binarized_image_device = copy_host_to_device(binarized_image_host)
    eroded_image_device = copy_host_to_device(eroded_image_host)
    filter_device = copy_host_to_device(filter_host)

    path = os.path.dirname(os.path.abspath(__file__))
    parameters = {
        'height_image': str(height_image), 
        'width_image': str(width_image),
        'height_filter': str(height_filter),
        'width_filter': str(width_filter)
        }
    kernel = kernel_creation(path, kernel_parameters = parameters)

    # Kernel excecution 
    module = compiler.SourceModule(kernel)
    erosion_function = module.get_function("erosion")
    erosion_function(
        binarized_image_device,
        filter_device,
        eroded_image_device, 
        block = (rows_device, columns_device, 1),
        grid = (100,100,1),
        ) 

    # Copy device variable to host device
    cuda.memcpy_dtoh(eroded_image_host, eroded_image_device)
    return eroded_image_host

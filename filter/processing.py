# Libraries
import numpy as np
import os
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda import compiler
from utils import copy_host_to_device, kernel_creation


def filter(gray_image, filter):
    # Definition of necessary variables 
    height_image, width_image = gray_image.shape
    height_filter, width_filter = filter.shape
    rows_device = round(width_image/100)
    columns_device = round(height_image/100)

    # Definition of necessary variables 
    gray_image_host = np.array(gray_image).astype(np.float32)   #uint32
    filter_host = np.array(filter) .astype(np.float32)
    filtered_image_host = np.zeros((height_image, width_image)).astype(np.float32)

    # Required memory allocation of the device, for variables
    gray_image_device, kernel_device = copy_host_to_device(gray_image_host, filter_host)
    image_filtered_device = copy_host_to_device(filtered_image_host)

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
    filter_function = module.get_function("filter") 
    filter_function(
        gray_image_device,
        kernel_device,
        image_filtered_device,
        block = (rows_device, columns_device, 1),
        grid = (100,100,1),
    )
    
    # Copy device variable to host device
    cuda.memcpy_dtoh(filtered_image_host, image_filtered_device)
    return filtered_image_host


def kernel_selection(name):
    option = name
  
    if option == "laplace":
        kernel = np.array([[1,1,1],
                           [1,-8,1],
                           [1,1,1]])
      
    elif option == "convolucion":
        kernel = np.array([[0.04,0.04,0.04,0.04,0.04],
                           [0.04,0.04,0.04,0.04,0.04],
                           [0.04,0.04,0.04,0.04,0.04],
                           [0.04,0.04,0.04,0.04,0.04],
                           [0.04,0.04,0.04,0.04,0.04]])
        
    elif option == "gauss_a":
        kernel = np.array([[1,2,1],
                           [2,4,2],
                           [1,2,1]])
        
    elif option == "gauss_b":
        kernel = np.array([[1,4,6,4,1],
                           [4,16,24,16,4],
                           [6,24,36,24,6],
                           [4,16,24,16,4],
                           [1,4,6,4,1]]) 
         
    elif option == "gauss_c":
        kernel = np.array([[0,0,0],
                           [1,1,1],
                           [0,0,0]]) 
         
    elif option == "gauss_d":
        kernel = np.array([[1,0,0],
                           [0,1,0],
                           [0,0,1]])  
   
    elif option == "prewitt_a":
        kernel = np.array([[-1,-1,-1],    
                           [0,0,0],
                           [1,1,1]])  
         
    elif option == "prewitt_b":
        kernel = np.array([[-1,0,1],
                           [-1,0,1],
                           [-1,0,1]])  
        
    elif option == "prewitt_c":
        kernel = np.array([[0,1,1],
                           [-1,0,1],
                           [-1,-1,0]])
      
    elif option == "prewitt_d":
        kernel = np.array([[1,1,1],
                           [0,0,0],
                           [-1,-1,-1]])
          
    elif option == "roberts_a":
        kernel = np.array([[-1,0],
                           [0,1]])
              
    elif option == "roberts_b":
        kernel = np.array([[-1,0],
                           [1,0]])  
   
    elif option == "roberts_c":
        kernel = np.array([[-1,1],
                           [0,0]])

    return kernel
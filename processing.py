# Importacion de librerias
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda import compiler


# Definicion del kernel
kernel_code_template_rgb2gray = """
__global__ void rgb2gray(unsigned int *grayImage, unsigned int *rgbImage)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x<%(width_image)s && y<%(height_image)s){

        int grayOffset = y * %(width_image)s + x;
        int rgbOffset = grayOffset * %(channels)s;

        unsigned int r = rgbImage[rgbOffset];
        unsigned int g = rgbImage[rgbOffset + 1]; 
        unsigned int b = rgbImage[rgbOffset + 2];

        grayImage[grayOffset] = int((r + g + b) / 3); 
    }
}
"""

kernel_code_template_gray2bin = """
__global__ void gray2bin( float *grayimage, float *binimage)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    if(tx<%(width_image)s && ty< %(height_image)s){  

        int offsetgray = ty * %(width_image)s + tx;

        if(grayimage[offsetgray]<%(threshold)s){

            binimage[offsetgray]=0;
        }

        else{
            binimage[offsetgray]=1;
        }
    }
}
"""

kernel_code_template_erosion = """
__global__ void erosion(unsigned int *image, unsigned int *kernel, unsigned int *imagefiltred)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
   
    int kernelRowRadius = (%(height_filter)s-1)/2;
    int kernelColRadius = (%(width_filter)s-1)/2;

    if(ty<%(height_image)s && tx<%(width_image)s){

        int startRow = ty - kernelRowRadius;
        int startColumn = tx - kernelColRadius;
        int exception = 0;

        for (int i=0; i<%(height_filter)s; i++){
            for (int j=0; j<%(width_filter)s; j++){

                int currentRow = startRow + i;
                int currentCol = startColumn + j;

                if(currentRow>=0 && currentRow<%(height_image)s && currentCol>=0 && currentCol<%(width_image)s){                  
                    if(kernel[i * %(height_filter)s +j]==1 && image[currentRow * %(width_image)s + currentCol]==0){
                        
                        exception =1;
                        imagefiltred[ty * %(width_image)s + tx] = 0;
                    }

                    else{
                        if(exception!=1 && i==(%(height_filter)s-1) && j==(%(width_filter)s-1)){
                            
                            imagefiltred[ty * %(width_image)s + tx] = 255;
                            exception = 0;  
                        }       
                    }
                }

                else{
                    imagefiltred[ty * %(width_image)s + tx] = 0;
                }                
            }
        } 
    }  
}
"""

kernel_code_template_dilatation = """
__global__ void dilatation(unsigned int *image, unsigned int *kernel, unsigned int *imagefiltred)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    int kernelRowRadius = (%(height_filter)s-1)/2;
    int kernelColRadius = (%(width_filter)s-1)/2;

    if(ty<%(height_image)s && tx<%(width_image)s){

        int startRow= ty - kernelRowRadius;
        int startColumn= tx - kernelColRadius;

        for (int i=0; i<%(height_filter)s; i++){
            for (int j=0; j<%(width_filter)s; j++){

                int currentRow = startRow + i;
                int currentCol = startColumn + j;

                if(currentRow>=0 && currentRow<%(height_image)s && currentCol>=0 && currentCol<%(width_image)s){                    
                    if(kernel[i * %(height_filter)s +j]==1 && image[currentRow * %(width_image)s + currentCol]==1){

                        imagefiltred[ty * %(width_image)s + tx] = 255;
                    }  
                }

                else{
                    imagefiltred[ty * %(width_image)s + tx] = 0;
                }
            }
        } 
    }  
}
"""

kernel_code_template_filter = """
__global__ void filter(float *image, float *kernel, float *imagefiltred)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    float accum=0;

    int kernelRowRadius = %(height_filter)s/2;
    int kernelColRadius = %(width_filter)s/2;

    if(ty < %(height_image)s && tx < %(width_image)s){

        int startRow= ty - kernelRowRadius;
        int startColumn= tx - kernelColRadius;

        for (int i=0; i<%(height_filter)s; i++){
            for (int j=0; j< %(width_filter)s; j++){

                int currentRow = startRow + i;
                int currentCol = startColumn + j;

                if(currentRow>=0 && currentRow < %(height_image)s && currentCol>=0 && currentCol < %(width_image)s){

                    accum+= image[(currentRow * %(width_image)s + currentCol)] * kernel[i * %(height_filter)s +j];
                }

                else{
                    accum=0;
                }
            }
        } 

        imagefiltred[ty * %(width_image)s + tx] = accum;
    }  
}
"""


def rgb2gray(rgb_image, height_image, width_image, channels=3):
    """
    Method for converting RGB to grayscale
    """
    # Definition of necessary variables 
    rows_device = round(width_image/100)
    columns_device = round(height_image/100)

    # Assignment of variables within the host
    image_rgb_host = np.array(rgb_image).astype(np.float32)
    image_gray_host = np.zeros((height_image, width_image)).astype(np.float32)

    # Required memory allocation of the device, for variables
    image_rgb_device = cuda.mem_alloc(image_rgb_host.nbytes)
    image_gray_device = cuda.mem_alloc(image_gray_host.nbytes)

    # Copy data from host to device 
    cuda.memcpy_htod(image_rgb_device, image_rgb_host)

    # Kernel with the necessary values 
    kernel_code = kernel_code_template_rgb2gray % {
        'width_image': str(width_image),
        'height_image': str(height_image),
        'channels': str(channels)
    }

    # Kernel call
    module = compiler.SourceModule(kernel_code)
    rgb2gray_function = module.get_function('rgb2gray')

    # Kernel execution 
    rgb2gray_function(
        # input
        image_gray_device,
        # output
        image_rgb_device, 
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (rows_device, columns_device, 1),
        grid = (100,100,1),
    )

    # Copy the result to the host, hosted on the device
    cuda.memcpy_dtoh(image_gray_host, image_gray_device)
    return image_gray_host


def gray2bin(gray_image, threshold):
    """
    Method for converting an image grayscale to binarized
    """
    # Definition of necessary variables 
    height_image, width_image = gray_image.shape
    rows_device = round(width_image/100)
    columns_device = round(height_image/100)

    # Assignment of variables within the host
    gray_image_host = np.array(gray_image).astype(np.float32)
    binarized_image_host = np.zeros((height_image, width_image)).astype(np.float32)

    # Required memory allocation of the device, for variables
    gray_image_device = cuda.mem_alloc(gray_image_host.nbytes)
    binarized_image_device = cuda.mem_alloc(binarized_image_host.nbytes)

    # Copy data from host to device 
    cuda.memcpy_htod(gray_image_device, gray_image_host)

    # Kernel with the necessary values 
    kernel_code = kernel_code_template_gray2bin % {
        'height_image': height_image, 
        'width_image': width_image,
        'threshold': threshold
    }

    # Kernel call 
    module = compiler.SourceModule(kernel_code)
    gray2bin_function = module.get_function('gray2bin')
    
    # Kernel execution 
    gray2bin_function(
        # input
        gray_image_device,
        # output
        binarized_image_device, 
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (rows_device, columns_device, 1),
        grid = (100,100,1),
    )

    # Copy the result to the host, hosted on the device
    cuda.memcpy_dtoh(binarized_image_host, binarized_image_device)
    return binarized_image_host


def erosion(binarized_image, filter):
    """
    Method of applying erosion to a binarized image
    """
    # Definition of necessary variables 
    height_image, width_image = binarized_image.shape
    height_filter, width_filter = filter.shape
    rows_device = round(width_image/100)
    columns_device = round(height_image/100)

    # Assignment of variables within the host
    binarized_image_host = np.array(binarized_image).astype(np.uint32)
    eroded_image_host = np.zeros((height_image, width_image)).astype(np.uint32)
    filter_host = np.array(filter).astype(np.uint32)

    # Required memory allocation of the device, for variables
    binarized_image_device = cuda.mem_alloc(binarized_image_host.nbytes)
    eroded_image_device = cuda.mem_alloc(eroded_image_host.nbytes)
    filter_device = cuda.mem_alloc(filter_host.nbytes)

    # Copy data from host to device 
    cuda.memcpy_htod(binarized_image_device, binarized_image_host)
    cuda.memcpy_htod(filter_device, filter_host)

    # Kernel with the necessary values 
    kernel_code = kernel_code_template_erosion % {
        'height_image': height_image, 
        'width_image': width_image,
        'height_filter': height_filter,
        'width_filter': width_filter
        }

    # Kernel call 
    module = compiler.SourceModule(kernel_code)
    erosion_function = module.get_function("erosion")

    # Kernel execution 
    erosion_function(
        # inputs
        binarized_image_device,
        filter_device,
        # output
        eroded_image_device, 
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (rows_device, columns_device, 1),
        grid = (100,100,1),
        ) 

    # Copy the result to the host, hosted on the device
    cuda.memcpy_dtoh(eroded_image_host, eroded_image_device)
    return eroded_image_host


def dilatation(binarized_image, filter):
    """
    Method of applying dilation to a binarized image 
    """
    # Definition of necessary variables 
    height_image, width_image = binarized_image.shape
    height_filter, width_filter = filter.shape
    rows_device = round(width_image/100)
    columns_device = round(height_image/100)

    # Assignment of variables within the host
    binarized_image_host = np.array(binarized_image).astype(np.uint32)
    dilated_image_host = np.zeros((height_image, width_image)).astype(np.uint32)
    filter_host = np.array(filter).astype(np.uint32)

    # Required memory allocation of the device, for variables
    binarized_image_device = cuda.mem_alloc(binarized_image_host.nbytes)
    dilatation_device = cuda.mem_alloc(dilated_image_host.nbytes)
    filter_device = cuda.mem_alloc(filter_host.nbytes)

    # Copy data from host to device 
    cuda.memcpy_htod(binarized_image_device, binarized_image_host)
    cuda.memcpy_htod(filter_device, filter_host)

    # Kernel with the necessary values 
    kernel_code = kernel_code_template_dilatation % {
        'height_image': height_image, 
        'width_image': width_image,
        'height_filter': height_filter,
        'width_filter': width_filter
        }

    # Kernel call
    module = compiler.SourceModule(kernel_code)
    dilatation_function = module.get_function("dilatation")

    # Kernel execution 
    dilatation_function(
        # inputs
        binarized_image_device,
        filter_device,
        #output
        dilatation_device, 
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (rows_device, columns_device, 1),
        grid = (100,100,1),
        ) 

    # Copy the result to the host, hosted on the device
    cuda.memcpy_dtoh(dilated_image_host, dilatation_device)
    return dilated_image_host


def filter(gray_image, filter):
    ''' 
    definition of the methods for the selection of the kernel and its respective application 
    '''
    # Definition of necessary variables 
    height_image, width_image = gray_image.shape
    height_filter, width_filter = filter.shape
    rows_device = round(width_image/100)
    columns_device = round(height_image/100)

    # Assignment of variables within the host
    gray_image_host = np.array(gray_image).astype(np.float32)
    filter_host = np.array(filter) .astype(np.float32)
    filtered_image_host = np.zeros((height_image, width_image)).astype(np.float32)

    # Required memory allocation of the device, for variables
    gray_image_device = cuda.mem_alloc(gray_image_host.nbytes)
    kernel_device = cuda.mem_alloc(filter_host.nbytes)
    image_filtered_device = cuda.mem_alloc(filtered_image_host.nbytes)

    # Copy data from host to device 
    cuda.memcpy_htod(gray_image_device, gray_image_host)
    cuda.memcpy_htod(kernel_device, filter_host)

    # Kernel with the necessary values 
    kernel_code = kernel_code_template_filter % {
    'height_image': height_image, 
    'width_image': width_image,
    'height_filter': height_filter,
    'width_filter': width_filter
    }

    # Kernel call
    module = compiler.SourceModule(kernel_code)
    filter_function = module.get_function('filter')

    # Kernel execution 
    filter_function(
        # inputs
        gray_image_device,
        kernel_device,
        #output
        image_filtered_device,
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads 
        block = (rows_device, columns_device, 1),
        grid = (100,100,1),
    )
    
    # Copy the result to the host, hosted on the device
    cuda.memcpy_dtoh(filtered_image_host, image_filtered_device)
    return filtered_image_host


def seleccion_kernel (nombre):
   option = nombre
  
   if option == "laplace":
      kernel= np.array([[1,1,1],
                        [1,-8,1],
                        [1,1,1]])
      
   elif option == "convolucion":
         kernel= np.array([[0.04,0.04,0.04,0.04,0.04],
                [0.04,0.04,0.04,0.04,0.04],
                [0.04,0.04,0.04,0.04,0.04],
                [0.04,0.04,0.04,0.04,0.04],
                [0.04,0.04,0.04,0.04,0.04],])
        
   elif option == "gauss_a":
         kernel= np.array([[1,2,1],
                           [2,4,2],
                           [1,2,1]])
        
   elif option == "gauss_b":
         kernel= np.array([[1,4,6,4,1],
                           [4,16,24,16,4],
                           [6,24,36,24,6],
                           [4,16,24,16,4],
                           [1,4,6,4,1]]) 
         
   elif option == "gauss_c":
         kernel= np.array([[0,0,0],
                           [1,1,1],
                           [0,0,0]]) 
         
   elif option == "gauss_d":
         kernel= np.array([[1,0,0],
                           [0,1,0],
                           [0,0,1]])  
   
   elif option == "prewitt_a":
         kernel= np.array([[-1,-1,-1],    
                           [0,0,0],
                           [1,1,1]])  
         
   elif option == "prewitt_b":
         kernel= np.array([[-1,0,1],
                           [-1,0,1],
                           [-1,0,1]])  
        
   elif option == "prewitt_c":
      kernel= np.array([[0,1,1],
                        [-1,0,1],
                        [-1,-1,0]])
      
   elif option == "prewitt_d":
      kernel= np.array([[1,1,1],
                        [0,0,0],
                        [-1,-1,-1]])
          
   elif option == "roberts_a":
      kernel= np.array([[-1,0],
                        [0,1]])
              
   elif option == "roberts_b":
      kernel= np.array([[-1,0],
                       [1,0]])  
   
   elif option == "roberts_c":
      kernel= np.array([[-1,1],
                        [0,0]])
   return kernel

    
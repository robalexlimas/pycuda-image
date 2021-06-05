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

        if(x < %(width_image)s && y < %(height_image)s) {

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

    if(tx<%(width)s && ty< %(height)s){  

        int offsetgray = ty * %(width)s + tx;

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



def rgb2gray(rgbimage, height_image, width_image, channels=3):
    """
    Method for converting RGB to grayscale 
    """
    # Assignment of variables within the host
    rgbimage_host = np.array(rgbimage).astype(np.uint32)
    grayimage_host = np.zeros((height_image, width_image)).astype(np.uint32)

    # Required memory allocation of the device, for variables
    rgbimage_device = cuda.mem_alloc(rgbimage_host.nbytes)
    grayimage_device = cuda.mem_alloc(grayimage_host.nbytes)

    # Copy data from host to device 
    cuda.memcpy_htod(rgbimage_device, rgbimage_host)

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
    rows_device = round(width_image/100)
    columns_device = round(height_image/100)

    rgb2gray_function(
        # input
        rgbimage_device,
        # outputs
        grayimage_device,
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads 
        block=(rows_device, columns_device, 1),
        grid = (100,100,1)
    )

    # Copy the result to the host, hosted on the device
    cuda.memcpy_dtoh(grayimage_host, grayimage_device)
    return grayimage_host


def erosion(binimage, height_image, width_image, filter, height_filter, width_filter):
    """
    Method of applying erosion to a binarized image
    """
    # Assignment of variables within the host
    binimage_host = np.array(binimage).astype(np.uint32)
    erosion_host = np.zeros((height_image, width_image)).astype(np.uint32)
    filter_host = np.array(filter).astype(np.uint32)

    # Required memory allocation of the device, for variables
    binimage_device = cuda.mem_alloc(binimage_host.nbytes)
    erosion_device = cuda.mem_alloc(erosion_host.nbytes)
    filter_device = cuda.mem_alloc(filter_host.nbytes)

    # Copy data from host to device 
    cuda.memcpy_htod(binimage_device, binimage_host)
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
    rows_device = round(width_image/100)
    columns_device = round(height_image/100)

    erosion_function(
        # inputs
        binimage_device,
        filter_device,
        # outputs
        erosion_device, 
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (rows_device, columns_device, 1),
        grid = (100,100,1),
        ) 

    # Copy the result to the host, hosted on the device
    cuda.memcpy_dtoh(erosion_host, erosion_device)
    return erosion_host


def dilatation(binimage, height_image, width_image, filter, height_filter, width_filter):
    """
    Method of applying dilation to a binarized image 
    """
    # Assignment of variables within the host
    binimage_host = np.array(binimage).astype(np.uint32)
    dilatation_host = np.zeros((height_image, width_image)).astype(np.uint32)
    filter_host = np.array(filter).astype(np.uint32)

    # Required memory allocation of the device, for variables
    binimage_device = cuda.mem_alloc(binimage_host.nbytes)
    dilatation_device = cuda.mem_alloc(dilatation_host.nbytes)
    filter_device = cuda.mem_alloc(filter_host.nbytes)

    # Copy data from host to device 
    cuda.memcpy_htod(binimage_device, binimage_host)
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
    rows_device = round(width_image/100)
    columns_device = round(height_image/100)

    dilatation_function(
        # inputs
        binimage_device,
        filter_device,
        #output
        dilatation_device, 
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (rows_device, columns_device, 1),
        grid = (100,100,1),
        ) 

    # Copy the result to the host, hosted on the device
    cuda.memcpy_dtoh(dilatation_host, dilatation_device)
    return dilatation_host


def gray2bin(image_gray, height, width, threshold):
    gray_cpu = np.array(image_gray).astype(np.float32)
    bin_cpu = np.zeros((height, width)).astype(np.float32)
    
    gray_gpu = cuda.mem_alloc(gray_cpu.nbytes)
    bin_gpu = cuda.mem_alloc(bin_cpu.nbytes)

    cuda.memcpy_htod(gray_gpu, gray_cpu)

    kernel_code = kernel_code_template_gray2bin % {
        'height': height, 
        'width': width,
        'threshold': threshold
    }

    mod = compiler.SourceModule(kernel_code)

    matrixmul = mod.get_function('gray2bin')
    matrixmul(
        # input
        bin_gpu,
        # output
        gray_gpu, 
        block=(6,36, 1),
        grid = (100,8,1)
    )

    cuda.memcpy_dtoh(bin_cpu, bin_gpu)
    return bin_cpu
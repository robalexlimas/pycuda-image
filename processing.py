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

        if(x < %(width)s && y < %(height)s) {

            int grayOffset = y * %(width)s + x;
            int rgbOffset = grayOffset * %(channels)s;

            unsigned int r = rgbImage[rgbOffset];
            unsigned int g = rgbImage[rgbOffset + 1]; 
            unsigned int b = rgbImage[rgbOffset + 2];

            grayImage[grayOffset] = int((r + g + b) / 3); 
        }
    }
"""

kernel_code_template_erosion = """
    __global__ void erosion(float *image, float *kernel, float *imagefiltred)
    {
        int tx = threadIdx.x + blockDim.x * blockIdx.x;
        int ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(ty < %(rows_image)s && tx < %(columns_image)s){
            int x = ((%(rows_image)s - %(rows_kernel)s) - 1);
            int y = ((%(columns_image)s - %(columns_kernel)s) - 1);
            for (int rows_image=0; rows_image<x; rows_image++){
                for (int columns_image=0; columns_image<y; columns_image++){
                    int check = 0;
                    for(int k=0; k<%(kernel_center_x)s; k++){
                        for (int l=0; l<%(kernel_center_y)s; l++){
                            if(kernel[k * %(kernel_center_x)s + l]==1 && image[((rows_image+k) * %(columns_image)s) + (columns_image+l)]==0){
                                check = 1;
                            } else {
                                if(check!=1 && k==2 && l==2){
                                    imagefiltred[((rows_image+1) * %(columns_image)s) + (columns_image+1)] = 1;    
                                }
                            } 
                        }
                    }
                }
            }
        }  
    }
"""


def erosion(image, kernel, height, width):
    """

    """
    # Asignacion de los tamanos de los vectores necesarios
    image_cpu = np.array(image).astype(np.uint32)
    erosion_cpu = np.zeros((height, width)).astype(np.uint32)
    kernel_cpu =  np.array(kernel).astype(np.uint32)

    # Asignacion de memoria requerida dentro del procesamiento
    image_gpu = cuda.mem_alloc(image_cpu.nbytes)
    erosion_gpu = cuda.mem_alloc(erosion_cpu.nbytes)
    kernel_gpu =  cuda.mem_alloc(kernel_cpu.nbytes)

    # Copia de la informacion a de la cpu a la gpu
    cuda.memcpy_htod(image_gpu, image_cpu)
    cuda.memcpy_htod(kernel_gpu, kernel_cpu)

    # Kernel modificado con los valores necesarios
    kernel_code = kernel_code_template_erosion % {
        'rows_image': str(height),
        'columns_image': str(width),
        'kernel_center_x': str(int((kernel.shape[1] - 1) / 2)),
        'kernel_center_y': str(int((kernel.shape[0] - 1) / 2)),
        'rows_kernel': str(kernel.shape[0]),
        'columns_kernel': str(kernel.shape[1])
    }  
    # LLamdo del kernel
    mod = compiler.SourceModule(kernel_code)
    erosion_function = mod.get_function('erosion')
    # Ejecucion del kernel
    erosion_function(
        image_gpu,
        kernel_gpu,
        erosion_gpu,
        block=(6,36, 1),
        grid = (100,8,1)
    )

    #Copia de los resultados procesados por el kernel al cpu
    cuda.memcpy_dtoh(erosion_cpu, erosion_gpu)
    return erosion_cpu


def rgb2gray(image, height, width, channels=3):
    """
    Metodo para la conversion de RGB a escala de grises
    """
    # Asignacion de los tamanos de los vectores necesarios
    image_cpu = np.array(image).astype(np.uint32)
    gray_cpu = np.zeros((height, width)).astype(np.uint32)

    # Asignacion de memoria requerida dentro del procesamiento
    image_gpu = cuda.mem_alloc(image_cpu.nbytes)
    gray_gpu = cuda.mem_alloc(gray_cpu.nbytes)

    # Copia de la informacion a de la cpu a la gpu
    cuda.memcpy_htod(image_gpu, image_cpu)

    # Kernel modificado con los valores necesarios
    kernel_code = kernel_code_template_rgb2gray % {
        'width': str(width),
        'height': str(height),
        'channels': str(channels)
    }

    # LLamdo del kernel
    mod = compiler.SourceModule(kernel_code)
    rgb2gray_function = mod.get_function('rgb2gray')
    # Ejecucion del kernel
    rgb2gray_function(
        gray_gpu,
        image_gpu, 
        block=(6,36, 1),
        grid = (100,8,1)
    )

    #Copia de los resultados procesados por el kernel al cpu
    cuda.memcpy_dtoh(gray_cpu, gray_gpu)
    return gray_cpu

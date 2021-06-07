# Importacion de librerias
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda import compiler

import os

from utils import copy_host_to_device, kernel_creation


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
    
    # Llamado del kernel
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

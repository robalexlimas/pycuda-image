# Importacion de librerias
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda import compiler

# Definicion del kernel
kernel_code_template = """
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

def rgb2gray(image, height, width, channels=3):
    """
    Metodo para la conversion de RGB a escala de grises
    """
    # Asignacion de los tamanos de los vectores necesarios
    a_cpu = np.array(image).astype(np.float32)
    b_cpu = np.zeros((height, width)).astype(np.float32)

    # Asignacion de memoria requerida dentro del procesamiento
    a_gpu = cuda.mem_alloc(a_cpu.nbytes)
    b_gpu = cuda.mem_alloc(b_cpu.nbytes)

    # Copia de la informacion a de la cpu a la gpu
    cuda.memcpy_htod(a_gpu, a_cpu)

    # Kernel modificado con los valores necesarios
    kernel_code = kernel_code_template % {
        'width': str(width),
        'height': str(height),
        'channels': str(channels)
    }

    # LLamdo del kernel
    mod = compiler.SourceModule(kernel_code)
    matrixmul = mod.get_function('rgb2gray')
    # Ejecucion del kernel
    matrixmul(
        b_gpu,
        a_gpu, 
        block=(6,36, 1),
        grid = (100,8,1)
    )

    #Copia de los resultados procesados por el kernel al cpu
    cuda.memcpy_dtoh(b_cpu, b_gpu)
    return b_cpu

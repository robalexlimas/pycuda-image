# Importacion de librerias
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda import compiler

# Definicion del kernel
kernel_code_template = """
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

def erosion(binimage, height_image, width_image, filter, height_filter, width_filter):
    """
    Metodo para la conversion de RGB a escala de grises
    """
    # Asignacion de los tamanos de los vectores necesarios
    binimage_cpu = np.array(binimage).astype(np.uint32)
    erosion_cpu = np.zeros((height_image, width_image)).astype(np.uint32)
    filter_cpu = np.array(filter).astype(np.uint32)

    # Asignacion de memoria requerida dentro del procesamiento
    binimage_gpu = cuda.mem_alloc(binimage_cpu.nbytes)
    erosion_gpu = cuda.mem_alloc(erosion_cpu.nbytes)
    filter_gpu = cuda.mem_alloc(filter_cpu.nbytes)

    # Copia de la informacion a de la cpu a la gpu
    cuda.memcpy_htod(binimage_gpu, binimage_cpu)
    cuda.memcpy_htod(filter_gpu, filter_cpu)

    # Kernel modificado con los valores necesarios
    kernel_code = kernel_code_template % {
        'height_image': height_image, 
        'width_image': width_image,
        'height_filter': height_filter,
        'width_filter': width_filter
        }

    # LLamdo del kernel
    mod = compiler.SourceModule(kernel_code)
    erosion = mod.get_function("erosion")

    # Ejecucion del kernel
    rows_gpu = round(width_image/100)
    columns_gpu = round(height_image/100)
    erosion(
        # inputs
        binimage_gpu,
        filter_gpu,
        #output
        erosion_gpu, 
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (rows_gpu, columns_gpu, 1),
        grid = (100,100,1),
        ) 

    #Copia de los resultados procesados por el kernel al cpu
    cuda.memcpy_dtoh(erosion_cpu, erosion_gpu)
    return erosion_cpu
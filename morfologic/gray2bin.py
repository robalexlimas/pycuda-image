import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda import compiler, gpuarray

kernel_code_template = """
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

def gray2bin(image_gray, height, width, threshold):
    gray_cpu = np.array(image_gray).astype(np.float32)
    bin_cpu = np.zeros((height, width)).astype(np.float32)
    
    gray_gpu = cuda.mem_alloc(gray_cpu.nbytes)
    bin_gpu = cuda.mem_alloc(bin_cpu.nbytes)

    cuda.memcpy_htod(gray_gpu, gray_cpu)

    kernel_code = kernel_code_template % {
        'height': height, 
        'width': width,
        'threshold': threshold
    }

    mod = compiler.SourceModule(kernel_code)

    matrixmul = mod.get_function('gray2bin')
    matrixmul(
        gray_gpu,
        bin_gpu, 
        block=(6,36, 1),
        grid = (100,8,1)
    )

    cuda.memcpy_dtoh(bin_cpu, bin_gpu)

    return bin_cpu

__global__ void filter(float *image, float *kernel, float *imagefiltred)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    int kernelRowRadius = $height_filter / 2;
    int kernelColRadius = $width_filter / 2;

    float accum = 0;

    if(ty < $height_image && tx < $width_image){

        int startRow = ty - kernelRowRadius;
        int startColumn = tx - kernelColRadius;

        for (int i = 0; i < $height_filter; i++){
            for (int j = 0; j < $width_filter; j++){

                int currentRow = startRow + i;
                int currentCol = startColumn + j;

                if(currentRow >= 0 && currentRow < $height_image && currentCol >= 0 && currentCol < $width_image){
                    
                    accum += image[(currentRow * $width_image + currentCol)] * kernel[i * $height_filter +j];
                }

                else{
                    accum = 0;
                }
            }
        } 
        imagefiltred[ty * $width_image + tx] = accum;
    }  
}
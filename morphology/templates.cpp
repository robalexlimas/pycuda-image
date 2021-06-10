__global__ void dilatation(unsigned int *image, unsigned int *kernel, unsigned int *imagefiltred)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    int kernelRowRadius = ($height_filter - 1) / 2;
    int kernelColRadius = ($width_filter - 1) / 2;

    if(ty < $height_image && tx < $width_image){

        int startRow = ty - kernelRowRadius;
        int startColumn = tx - kernelColRadius;

        for (int i = 0; i < $height_filter; i++){
            for (int j = 0; j < $width_filter; j++){

                int currentRow = startRow + i;
                int currentCol = startColumn + j;
                
                if(currentRow >= 0 && currentRow < $height_image && currentCol >= 0 && currentCol < $width_image){                    
                    if(kernel[i * $height_filter + j] == 1 && image[currentRow * $width_image + currentCol] == 1){
                        
                        imagefiltred[ty * $width_image + tx] = 255;
                    }  
                }

                else{
                    imagefiltred[ty * $width_image + tx] = 0;
                }
            }
        } 
    }  
}


__global__ void erosion(unsigned int *image, unsigned int *kernel, unsigned int *imagefiltred)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
   
    int kernelRowRadius = ($height_filter - 1) / 2;
    int kernelColRadius = ($width_filter - 1) / 2;

    if(ty < $height_image && tx < $width_image){

        int startRow = ty - kernelRowRadius;
        int startColumn = tx - kernelColRadius;

        int exception = 0;
        
        for (int i = 0; i < $height_filter; i++){
            for (int j = 0; j < $width_filter; j++){

                int currentRow = startRow + i;
                int currentCol = startColumn + j;
                
                if(currentRow >= 0 && currentRow < $height_image && currentCol >= 0 && currentCol < $width_image){                  
                    if(kernel[i * $height_filter +j] == 1 && image[currentRow * $width_image + currentCol] == 0){
                        
                        exception = 1;
                        imagefiltred[ty * $width_image + tx] = 0;
                    }

                    else{
                        if(exception != 1 && i == ($height_filter - 1) && j == ($width_filter - 1)){
                            
                            imagefiltred[ty * $width_image + tx] = 255;
                            exception = 0;  
                        }       
                    }
                }
                
                else{
                    imagefiltred[ty * $width_image + tx] = 0;
                }                
            }
        } 
    }  
}
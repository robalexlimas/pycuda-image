__global__ void dilatation(unsigned int *image, unsigned int *kernel, unsigned int *imagefiltred)
{
    int global_index_thead_x = threadIdx.x + blockDim.x * blockIdx.x;
    int global_index_thead_y = threadIdx.y + blockDim.y * blockIdx.y;

    int kernel_Row_Radius = ($height_filter - 1) / 2;
    int kernel_Column_Radius = ($width_filter - 1) / 2;

    if(global_index_thead_y < $height_image && global_index_thead_x < $width_image){

        int start_Column = global_index_thead_x - kernel_Column_Radius;
        int start_Row = global_index_thead_y - kernel_Row_Radius;

        for (int index_Row = 0; index_Row < $height_filter; index_Row++){
            
            for (int index_Column = 0; index_Column < $width_filter; index_Column++){

                int current_Column = start_Column + index_Column;
                int current_Row = start_Row + index_Row;
                
                if(current_Row >= 0 && current_Row < $height_image && current_Column >= 0 && current_Column < $width_image){                    
                    
                    if(kernel[index_Row * $height_filter + index_Column] == 1 && image[current_Row * $width_image + current_Column] == 1){
                        
                        imagefiltred[global_index_thead_y * $width_image + global_index_thead_x] = 255;
                    }  
                }

                else{

                    imagefiltred[global_index_thead_y * $width_image + global_index_thead_x] = 0;
                }
            }
        } 
    }  
}


__global__ void erosion(unsigned int *image, unsigned int *kernel, unsigned int *imagefiltred)
{
    int global_index_thead_x = threadIdx.x + blockDim.x * blockIdx.x;
    int global_index_thead_y = threadIdx.y + blockDim.y * blockIdx.y;
   
    int kernelRowRadius = ($height_filter - 1) / 2;
    int kernelColRadius = ($width_filter - 1) / 2;

    if(global_index_thead_y < $height_image && global_index_thead_x < $width_image){

        int start_Column = global_index_thead_x - kernelColRadius;
        int start_Row = global_index_thead_y - kernelRowRadius;

        int exception = 0;
        
        for (int index_Row = 0; index_Row < $height_filter; index_Row++){

            for (int index_Column = 0; index_Column < $width_filter; index_Column++){

                int current_Column = start_Column + index_Column;
                int current_Row = start_Row + index_Row;
                
                if(current_Row >= 0 && current_Row < $height_image && current_Column >= 0 && current_Column < $width_image){                  
                   
                    if(kernel[index_Row * $height_filter + index_Column] == 1 && image[current_Row * $width_image + current_Column] == 0){
                        
                        exception = 1;
                        imagefiltred[global_index_thead_y * $width_image + global_index_thead_x] = 0;
                    }

                    else{
                        
                        if(exception != 1 && index_Row == ($height_filter - 1) && index_Column == ($width_filter - 1)){
                            
                            exception = 0; 
                            imagefiltred[global_index_thead_y * $width_image + global_index_thead_x] = 255; 
                        }       
                    }
                }
                
                else{

                    imagefiltred[global_index_thead_y * $width_image + global_index_thead_x] = 0;
                }                
            }
        } 
    }  
}

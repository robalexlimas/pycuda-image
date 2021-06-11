__global__ void filter(float *image, float *kernel, float *imagefiltred)
{
    int global_index_thead_x = threadIdx.x + blockDim.x * blockIdx.x;
    int global_index_thead_y = threadIdx.y + blockDim.y * blockIdx.y;

    int kernel_Row_Radius = $height_filter / 2;
    int kernel_Column_Radius = $width_filter / 2;

    float accumulator = 0;

    if(global_index_thead_y < $height_image && global_index_thead_x < $width_image){

        int start_Row = global_index_thead_y - kernel_Row_Radius;
        int start_Column = global_index_thead_x - kernel_Column_Radius;

        for (int index_Row = 0; index_Row < $height_filter; index_Row++){
            
            for (int index_Column = 0; index_Column < $width_filter; index_Column++){

                int current_Row = start_Row + index_Row;
                int current_Column = start_Column + index_Column;

                if(current_Row >= 0 && current_Row < $height_image && current_Column >= 0 && current_Column < $width_image){
                    
                    accumulator += image[(current_Row * $width_image + current_Column)] * kernel[index_Row * $height_filter + index_Column];
                }

                else{

                    accumulator = 0;
                }
            }
        }
         
        imagefiltred[global_index_thead_y * $width_image + global_index_thead_x] = accumulator;
    }  
}

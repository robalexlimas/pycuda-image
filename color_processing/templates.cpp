__global__ void rgb2gray(unsigned int *grayImage, unsigned int *rgbImage)
    {
    int global_index_thead_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_index_thead_y = threadIdx.y + blockIdx.y * blockDim.y;

    if(global_index_thead_x < $width_image && global_index_thead_y < $height_image) {

        int gray_Offset = global_index_thead_y * $width_image + global_index_thead_x;
        int rgb_Offset = gray_Offset * $channels;

        unsigned int r = rgbImage[rgb_Offset];
        unsigned int g = rgbImage[rgb_Offset + 1]; 
        unsigned int b = rgbImage[rgb_Offset + 2];

        grayImage[gray_Offset] = int((r + g + b) / 3); 
    }
}


__global__ void gray2bin( float *grayimage, float *binimage)
{
    int global_index_thead_x = threadIdx.x + blockDim.x * blockIdx.x;
    int global_index_thead_y = threadIdx.y + blockDim.y * blockIdx.y;

    if(global_index_thead_x < $width_image && global_index_thead_y < $height_image){  

        int gray_Offset = global_index_thead_y * $width_image + global_index_thead_x;

        if(grayimage[gray_Offset] < $threshold){

            binimage[gray_Offset] = 0;
        }
        
        else{
            
            binimage[gray_Offset] = 1;
        }
    }
}

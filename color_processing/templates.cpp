__global__ void rgb2gray(unsigned int *grayImage, unsigned int *rgbImage)
    {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < $width && y < $height) {

        int grayOffset = y * $width + x;
        int rgbOffset = grayOffset * $channels;

        unsigned int r = rgbImage[rgbOffset];
        unsigned int g = rgbImage[rgbOffset + 1]; 
        unsigned int b = rgbImage[rgbOffset + 2];

        grayImage[grayOffset] = int((r + g + b) / 3); 
    }
}


__global__ void gray2bin( float *grayimage, float *binimage)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    if(tx < $width && ty < $height){  

        int offsetgray = ty * $width + tx;

        if(grayimage[offsetgray] < $threshold){

            binimage[offsetgray] = 0;
        }
        else{
            binimage[offsetgray] = 1;
        }
    }
}
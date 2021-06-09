# Libraries
import numpy as np

from image import array2image, array2vector_rgb, load_image, save_image
from color_processing.processing import rgb2gray


def main():
    # Files names
    input_name = 'shingeki.jpeg'
    output_name = 'shingeki_gray.jpeg'

    input_image = load_image(input_name)
    # Convert format input image to vector
    image = np.array(
        input_image.getdata()).reshape(input_image.size[1], input_image.size[0], 3
    )
    image_vector = array2vector_rgb(image)
    # Image size
    width, height = input_image.size[0], input_image.size[1]

    output_image_array = rgb2gray(image_vector, width, height)
    output_image = array2image(output_image_array)
    save_image(output_image, output_name)


if __name__=='__main__':
    main()

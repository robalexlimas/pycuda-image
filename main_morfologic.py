import numpy as np

# Importacion de librerias personalizadas
from image import array2image, array2vector_rgb, array2vector_gray, load_image, save_image 
from color_processing.processing import rgb2gray, gray2bin
from morphology.processing import dilatation


def main():
    # Files names
    input_name = 'shingeki.jpeg'
    output1_name = 'shingeki_gray.jpeg'
    output2_name = 'shingeki_binared.jpeg'
    output3_name = 'shingeki_dilated.jpeg'
    output4_name = 'shingeki_erosioned.jpeg'

    input_image = load_image(input_name)

    # Convert format input image to vector
    image = np.array(
        input_image.getdata()).reshape(input_image.size[1], input_image.size[0], 3
    )
    image_vector = array2vector_rgb(image)

    # Image size
    width, height = input_image.size[0], input_image.size[1]


    output_rgb2gray = rgb2gray(image_vector, width, height)

    threshold = 150
    output_binarized = gray2bin(output_rgb2gray, threshold)

    filter=np.array([[0,0,0,0,0],
                    [0,0,0,0,0],
                    [1,1,1,1,1],
                    [0,0,0,0,0],
                    [0,0,0,0,0]])


    output_dilatation = dilatation(output_binarized, filter)
    #output_erosion = erosion(output_binarized, filter)

    # Conversion del array resultante a formato imagen
    output1_image = array2image(output_rgb2gray)
    output2_image = array2image(output_binarized)
    output3_image = array2image(output_dilatation)
    #output4_image = array2image(output_erosion)
    
    # Guardado de la imagen procesada
    save_image(output1_image, output1_name)
    save_image(output2_image, output2_name)
    save_image(output3_image, output3_name)
    #save_image(output4_image, output4_name)


if __name__=='__main__':
    main()

import numpy as np

# Importacion de librerias personalizadas
from image import array2image, array2vectorgray, load_image, save_image
from gray2bin import gray2bin
from dilatation import dilatation
from erosion import erosion


def main():
    # Nombres de los archivos a procesar
    input_name = 'shingeki_gray.jpeg'
    output1_name = 'shingeki_binared.jpeg'
    output2_name = 'shingeki_dilated.jpeg'
    output3_name = 'shingeki_erosioned.jpeg'

    # Carga de la imagen de entrada
    input_image = load_image(input_name)

    # Conversion de la imagen de entrada a vector
    image = np.array(
        input_image.getdata()).reshape(input_image.size[1], input_image.size[0], 3
    )
    image_gray = image[:,:,0]
    image_grayscale = array2vectorgray(image_gray)

    # Tamano de la imagen original
    height_img, width_img = image_gray.shape

    # Conversion a escala de grises
    threshold = 150
    output_binarized = gray2bin(image_grayscale, height_img, width_img, threshold)

    filter=np.array([[0,0,0,0,0],
                    [0,0,0,0,0],
                    [1,1,1,1,1],
                    [0,0,0,0,0],
                    [0,0,0,0,0]])
    height_fil, width_fil = filter.shape

    # Dilatacion y erosion
    output_dilatation = dilatation(output_binarized, height_img, width_img, filter, height_fil, width_fil)
    output_erosion = erosion(output_binarized, height_img, width_img, filter, height_fil, width_fil)

    # Conversion del array resultante a formato imagen
    output1_image = array2image(output_binarized)
    output2_image = array2image(output_dilatation)
    output3_image = array2image(output_erosion)
    # Guardado de la imagen procesada
    save_image(output1_image, output1_name)
    save_image(output2_image, output2_name)
    save_image(output3_image, output3_name)


if __name__=='__main__':
    main()

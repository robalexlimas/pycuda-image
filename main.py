import numpy as np

# Importacion de librerias personalizadas
from image import array2image, array2vector, load_image, save_image
from processing import rgb2gray, erosion


def main():
    # Nombres de los archivos a procesar
    input_name = 'shingeki.jpeg'
    output_name = 'shingeki_gray.jpeg'
    """
    # Carga de la imagen de entrada
    input_image = load_image(input_name)
    # Conversion de la imagen de entrada a vector
    image = np.array(
        input_image.getdata()).reshape(input_image.size[1], input_image.size[0], 3
    )
    image_vector = array2vector(image)
    # Tamano de la imagen original
    height, width = input_image.size[1], input_image.size[0]

    # Conversion a escala de grises
    output_image_array = rgb2gray(image_vector, height, width)
    # Conversion del array resultante a formato imagen
    output_image = array2image(output_image_array)
    # Guardado de la imagen procesada
    save_image(output_image, output_name)
    """
    image = np.array([[0,0,1,0,0],
                      [1,1,1,1,0],
                      [0,0,1,0,0],
                      [1,1,1,1,1],
                      [0,0,1,0,0],
                      [0,0,1,0,0]])
    kernel = np.array([[0,0,0],
                       [1,1,1],
                       [0,0,0]])
    t = erosion(image, kernel, image.shape[0], image.shape[1])
    print(t)


if __name__=='__main__':
    main()

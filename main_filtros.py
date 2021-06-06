import numpy as np

from image import array2image,load_image, array2vectorgray,save_image
from processing import filter, seleccion_kernel
import matplotlib.image as img
import matplotlib.pyplot as plt


def main():
    '''llamada de las imagenes a procesar y asignación de nombre a la imagen de salida  '''
    input_name = 'shingeki_gray.jpeg'
    output_name = 'shingeki_conv.jpeg'

    '''extracción de las dimenciones de la imagen a procesar
    y conversión de la imagen a array '''
    input_image = load_image(input_name)
    image = np.array(
        input_image.getdata()).reshape(input_image.size[1], input_image.size[0], 3
    )

    print(image.shape)

    ''' extraccióm de un solo canal de la imagen en escala de grices  '''
    image_gray=image[:,:,0]
    
    print(image_gray.shape)
    '''conversion del aray de la imagen a tipo vector '''
    image_vector = array2vectorgray(image_gray)

    '''extracción de las dimenciones de la imagen a procesar
    dimensiones delñ alto y ancho de la imagen 
    '''

    height, width = image_gray.shape

    print(height)
    print(width)

    '''selección del kernel o filtro a aplicar a la imagen '''
    kernel=seleccion_kernel("laplace")
    print(kernel)
    print(kernel.shape)

    '''aplicación del filtro a la imagen'''
    output_image_array = filter(image_gray,kernel)

    '''conversión del array a imagen para guardar la imagen procesada '''
    output_image = array2image(output_image_array)

    ''' funcion de almacenamiento de la imagen procesada '''

    print(output_image, input_image)
    save_image(output_image, output_name)
    '''plot de la imagen para mostrar en pantalla los resultados obtenidos '''

    plt.imshow(output_image_array,cmap='gray')
    plt.show()
    

if __name__=='__main__':
    main()
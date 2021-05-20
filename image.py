import numpy as np
import os

from PIL import Image


def load_image(filename):
    """
    Metodo para la carga de la imagen
    """
    base_dir = os.getcwd()
    folder_images = 'images'
    path = os.path.join(base_dir, folder_images)
    image_path = os.path.join(path, filename)
    return Image.open(image_path)


def array2vector(image):
    """
    Metodo para convertir imagen de entrada a un vector
    """
    height, width, channels = image.shape
    image_array_list = []

    for row in range(height):
        for column in range(width):
            for channel in range(channels):
                image_array_list.append(image[row][column][channel])

    return np.array(image_array_list)


def array2image(image_array):
    """
    Metodo para convertir una matriz a imagen
    """
    image = Image.fromarray(image_array)
    return image.convert('RGB')


def save_image(image, filename):
    """
    Metodo para guardar la imagen
    """
    base_dir = os.getcwd()
    folder_images = 'images'
    path = os.path.join(base_dir, folder_images)
    if not (os.path.exists(path)):
        os.mkdir(path)
    image_path = os.path.join(path, filename)
    image.save(image_path)

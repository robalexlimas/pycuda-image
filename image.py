import numpy as np
import os

from PIL import Image


def load_image(filename, folder="images"):
    base_dir = os.getcwd()
    path = os.path.join(base_dir, folder)
    image_path = os.path.join(path, filename)
    return Image.open(image_path)


def array2vector(image):
    height, width, channels = image.shape
    image_array_list = []

    for row in range(height):
        for column in range(width):
            for channel in range(channels):
                image_array_list.append(image[row][column][channel])

    return np.array(image_array_list)


def array2image(image_array):
    image = Image.fromarray(image_array)
    return image.convert('RGB')


def save_image(image, filename, folder="images"):
    base_dir = os.getcwd()
    path = os.path.join(base_dir, folder)
    if not (os.path.exists(path)):
        os.mkdir(path)
    image_path = os.path.join(path, filename)
    image.save(image_path)

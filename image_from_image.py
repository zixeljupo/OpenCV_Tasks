import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import numpy.random as random
import cv2
import tensorflow as tf
from time import time
from collections import OrderedDict
from glob import glob
import os

print(cv2.__version__)


def read_image(path, img_format: str):

    if img_format == 'rgb':
        im = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    elif img_format == 'gray':
        im = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
    else:
        im = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im


def create_image_proto(image, stride):
    image_slices = np.array(np.split(image, stride, axis=0), dtype=image.dtype)
    image_slices = np.array(np.split(image_slices, stride, axis=2), dtype=image.dtype)
    image_means = np.mean(image_slices, axis=(2, 3))
    return image_means


if __name__ == "__main__":

    palette_folder = "C:/Users/Timofey/Desktop/OPEN_CV/Data/cats/"
    image_path = "C:/Users/Timofey/Desktop/OPEN_CV/Data/woman.jpg"

    box_size = 1
    result_size = 6000
    data_size = 60

    palette_image_paths = glob(os.path.join(palette_folder, '*'))

    image = read_image(image_path, "rgb")
    image = cv2.resize(image, (result_size, result_size))

    palette_image_list = []
    for palette_image_path in palette_image_paths:

        palette_image = read_image(palette_image_path, "rgb")
        palette_image = cv2.resize(palette_image, (data_size, data_size))
        palette_image_list.append(palette_image)

        print(palette_image_path, np.shape(palette_image))

    palette_images = np.array(palette_image_list, dtype=np.uint8)
    palette_means = np.mean(palette_image_list, axis=(1, 2))

    proto = create_image_proto(image, result_size//data_size)
    distances = np.expand_dims(proto, axis=2) - np.expand_dims(np.expand_dims(palette_means, 0), 0)
    distances = np.abs(distances)
    distances = np.sum(distances, axis=3)
    indices = np.argmin(distances, axis=2)

    result_image = np.zeros_like(image)
    for i in range(result_size//data_size):
        for j in range(result_size//data_size):
            result_image[i*data_size:i*data_size+data_size, \
                         j*data_size:j*data_size+data_size] = palette_images[indices[j, i]]

    result_image = cv2.blur(result_image, (3, 3))

    cv2.imwrite("./test.jpg", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

    plt.figure("Test")
    plt.subplot(121)
    plt.title("Source")
    plt.imshow(image)

    plt.subplot(122)
    plt.title("Catifyed")
    plt.imshow(result_image)
    plt.show()

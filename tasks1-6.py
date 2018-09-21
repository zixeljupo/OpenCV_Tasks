import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import numpy.random as random
import cv2
import tensorflow as tf
from time import time
from collections import OrderedDict

print(cv2.__version__)


def read_image(path, img_format: str):

    im = cv2.imread(path)

    if img_format == 'rgb':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if img_format == 'gray' and len(np.shape(im)) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    return im


def linear_correction(im, a=0.0, b=0.0):
    return np.clip(im.astype(np.float32)*a + b, 0.0, 255.0).astype(np.uint8)


def gamma_correction(im, gamma=1.0):
    return (np.clip(np.power(im.astype(np.float32)/255.0, gamma), 0.0, 255.0)*255.0).astype(np.uint8)


def build_hist(im):

    im = im.astype(np.int64)

    c = 1 if len(np.shape(im)) == 2 else np.shape(im)[2]
    if len(np.shape(im)) == 3:
        im = np.sum(im, axis=2)

    brightness = (im // c).reshape(im.size)
    hist = np.histogram(brightness, bins=256, range=(-0.5, 255.5))
    return hist


def image_gradients(im, kernel_size):

    grad_x = (cv2.Sobel(im, ddepth=-1, dx=1, dy=0, ksize=kernel_size)).astype(np.float32)
    grad_y = (cv2.Sobel(im, ddepth=-1, dx=0, dy=1, ksize=kernel_size)).astype(np.float32)

    modulus = np.expand_dims(np.sqrt(grad_x * grad_x + grad_y * grad_y) / (127.5 * np.sqrt(2)), axis=3)
    direction = (np.arctan2(grad_x - 127.5, grad_y - 127.5)/np.pi + 1.0) / 2.0
    colormap = plt.get_cmap("hsv")

    direction_image = colormap(direction).astype(np.float32)
    dir_mod_image = modulus * direction_image * 255.0
    result = dir_mod_image[:, :, 0:3]

    return result.astype(np.uint8)


def show_colormap(name="hsv"):

    colormap = plt.get_cmap(name)
    values = np.tile(np.arange(0, 360), (30, 1))/360.0
    image = colormap(values)

    plt.figure("Colormap")
    plt.imshow(image)


def task1():

    path = "C:/Users/Timofey/Desktop/OPEN_CV/Data/nsu.jpg"
    intensity = 200

    image = read_image(path, "rgb")
    print(np.shape(image))

    brightness = np.sum(image.astype(np.int64), axis=2)
    brightness = brightness.astype(np.float32)/3.0
    good_pixels = brightness > intensity
    print(good_pixels.astype(np.int64).sum())

    plt.title("Source image")
    plt.imshow(image)
    plt.show()


def task2():
    path = "C:/Users/Timofey/Desktop/OPEN_CV/Data/nsu.jpg"
    centers = [(552, 104), (346, 126)]

    image = read_image(path, "rgb")
    print(np.shape(image))

    for center in centers:
        cv2.ellipse(image, center, (43, 25), 0.0, 0.0, 360.0, (255, 0, 255), 3)

    plt.title("Source image with ellipses!")
    plt.imshow(image)
    plt.show()


def task3():

    path = "C:/Users/Timofey/Desktop/OPEN_CV/Data/gg.jpg"
    image = read_image(path, "rgb")
    print(np.shape(image))

    lc_image = linear_correction(image, a=1.0, b=50.0)
    gc_image = gamma_correction(image, gamma=0.5)
    image_hist = build_hist(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    eq_image = cv2.equalizeHist(image)
    grayscale_hist = build_hist(image)
    equalized_hist = build_hist(eq_image)

    plt.subplot(241)
    plt.title("Source image")
    imgplot = plt.imshow(image)
    imgplot.set_cmap('gray')

    plt.subplot(242)
    plt.title("LC")
    imgplot = plt.imshow(lc_image)
    imgplot.set_cmap('gray')

    plt.subplot(243)
    plt.title("GC")
    imgplot = plt.imshow(gc_image)
    imgplot.set_cmap('gray')

    plt.subplot(244)
    plt.title("Source image hist")
    plt.plot(image_hist[0])

    plt.subplot(245)
    plt.title("Source image in grayscale")
    imgplot = plt.imshow(image)
    imgplot.set_cmap('gray')

    plt.subplot(246)
    plt.title("Equalized image")
    imgplot = plt.imshow(eq_image)
    imgplot.set_cmap('gray')

    plt.subplot(247)
    plt.title("Grayscale image hist")
    plt.plot(grayscale_hist[0])

    plt.subplot(248)
    plt.title("Equalized hist")
    plt.plot(equalized_hist[0])

    plt.show()


def task4():

    path = "C:/Users/Timofey/Desktop/OPEN_CV/Data/church.jpg"
    image = read_image(path, "rgb")
    print(np.shape(image))

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    filtered_images = OrderedDict()
    filtered_images["Gaussian"] = cv2.GaussianBlur(image, (15, 15), 0)
    filtered_images["Median"] = cv2.medianBlur(image, 15)
    filtered_images["Laplace"] = cv2.Laplacian(gray_image, ddepth=-1)
    filtered_images["Sobel 1"] = cv2.Sobel(gray_image, ddepth=-1, dx=1, dy=1, ksize=5)
    filtered_images["Sobel 2"] = cv2.Sobel(gray_image, ddepth=-1, dx=1, dy=0, ksize=3)
    filtered_images["Sobel 3"] = cv2.Sobel(gray_image, ddepth=-1, dx=0, dy=1, ksize=3)
    filtered_images["Sobel 4"] = cv2.Sobel(gray_image, ddepth=-1, dx=1, dy=1, ksize=1)

    gradients = image_gradients(gray_image, kernel_size=3)

    show_colormap(name="hsv")

    plt.figure("Filter comparision")

    plt.subplot(331)
    plt.title("My image")
    plt.imshow(image)

    i = 0
    for k, v in filtered_images.items():
        plt.subplot(3, 3, i + 2)
        plt.title(k)
        plt.imshow(v)
        print(k + " " + str(np.shape(v)))
        i += 1

    show_colormap(name="hsv")

    plt.figure("Gradients visualization")

    plt.subplot(121)
    plt.title("Source image")
    plt.imshow(image)

    plt.subplot(122)
    plt.title("Gradients")
    plt.imshow(gradients)

    plt.figure("Noise reduction")

    mean = 0
    stddev = 50
    uniform_range = 50
    gauss_kernel_size = 7
    median_kernel_size = 7

    im1 = image.copy().astype(np.float64)
    im2 = image.copy().astype(np.float64)

    im1 += np.random.normal(mean, stddev, image.shape)
    im2 += np.random.uniform(-uniform_range, uniform_range + 1, image.shape)
    im1 = np.clip(im1, 0.0, 255.0)
    im2 = np.clip(im2, 0.0, 255.0)
    im1 = im1.astype(np.uint8)
    im2 = im2.astype(np.uint8)

    plt.subplot(231)
    plt.title("Gaussian noise")
    plt.imshow(im1)

    plt.subplot(234)
    plt.title("Uniform noise")
    plt.imshow(im2)

    plt.subplot(232)
    plt.title("Gaussian noise + gauss filter")
    plt.imshow(cv2.GaussianBlur(im1, (gauss_kernel_size, gauss_kernel_size), 0))

    plt.subplot(235)
    plt.title("Uniform noise + gauss filter")
    plt.imshow(cv2.GaussianBlur(im2, (gauss_kernel_size, gauss_kernel_size), 0))

    plt.subplot(233)
    plt.title("Gaussian noise + median filter")
    plt.imshow(cv2.medianBlur(im1, median_kernel_size))

    plt.subplot(236)
    plt.title("Uniform noise + median filter")
    plt.imshow(cv2.medianBlur(im2, median_kernel_size))

    plt.show()


def task5():

    path = "C:/Users/Timofey/Desktop/OPEN_CV/Data/shsh.jpg"
    image = read_image(path, "gray")
    blur = cv2.GaussianBlur(image, (3, 3), 0)

    threshold_value = 70
    _, result = cv2.threshold(image, threshold_value, 255, 0)

    _, th1 = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 71, -30)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 71, -30)
    titles = ['Original Image', 'Global Thresholding',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [blur, th1, th2, th3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    path = "C:/Users/Timofey/Desktop/OPEN_CV/Data/shsh.jpg"
    image = read_image(path, "gray")

    ret1, th1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    images = [image, 0, th1,
              image, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()


def task6():

    path = "C:/Users/Timofey/Desktop/OPEN_CV/Data/wheel.jpg"
    image = read_image(path, "gray")

    ratio = 3
    kernel_size = 3
    threshold = 100

    canny = cv2.blur(image, (3, 3))
    canny = cv2.Canny(canny, threshold, threshold*ratio, kernel_size)

    _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    drawing = np.zeros((canny.shape[0], canny.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = random.uniform(0, 256, 3)
        cv2.drawContours(drawing, contours, i, color, 1, cv2.LINE_8, hierarchy, 0)

    plt.subplot(131)
    plt.title("Source")
    plt.imshow(image, cmap="gray")

    plt.subplot(132)
    plt.title("Edges")
    plt.imshow(canny != 0, cmap="gray")

    plt.subplot(133)
    plt.title("Edges")
    plt.imshow(drawing, cmap="gray")

    plt.show()


if __name__ == "__main__":

    task1()
    task2()
    task3()
    task4()
    task5()
    task6()

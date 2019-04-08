__author__ = 'crazyj'

import numpy as np
import cv2
import matplotlib.pyplot as plt

def thresholding():
    img = cv2.imread('image/girl02.PNG', cv2.IMREAD_GRAYSCALE)

    ret, thr1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thr2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    blur = cv2.GaussianBlur(img, (5,5), 0)
    ret, thr3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    titles = [ 'org', 'histogram', 'g-thresholding',
               'org', 'histogram', 'otsu',
               'gaussian', 'histogram', 'otsu']
    images= [img, 0, thr1, img, 0, thr2, blur, 0, thr3]

    for i in range(3):
        plt.subplot(3,3, i*3+1), plt.imshow(images[i*3], 'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

        plt.subplot(3,3, i*3+2), plt.hist(images[i*3].ravel(), 256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])

        plt.subplot(3,3, i*3+3), plt.imshow(images[i*3+2], 'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()

thresholding()



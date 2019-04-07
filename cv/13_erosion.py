__author__ = 'crazyj'

import numpy as np
import cv2


def morph():
    img = cv2.imread('image/alphabet.jpg', cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((3,3), np.uint8)

    erosion = cv2.erode(img, kernel, iterations=1)
    dilation = cv2.dilate(img, kernel, iterations=1)

    cv2.imshow('org', img)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilation', dilation)

    # opening
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)

morph()
cv2.waitKey(0)
cv2.destroyAllWindows()


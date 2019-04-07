__author__ = 'crazyj'

import numpy as np
import cv2
import matplotlib.pyplot as plt

def canny():

    img = cv2.imread('image/girl02.png', cv2.IMREAD_GRAYSCALE)

    edge1 = cv2.Canny(img, 50, 200)
    edge2 = cv2.Canny(img, 100, 200)
    edge3 = cv2.Canny(img, 170, 200)

    cv2.imshow('org', img)
    cv2.imshow('canny edge1', edge1)
    cv2.imshow('canny edge2', edge2)
    cv2.imshow('canny edge3', edge3)


canny()

cv2.waitKey(0)
cv2.destroyAllWindows()

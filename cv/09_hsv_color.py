__author__ = 'crazyj'

import numpy as np
import cv2


def hsv():
    # BGR 색을 HSV 색 포맷으로 변환
    blue=np.uint8([[[255,0,0]]])
    green=np.uint8([[[0,255,0]]])
    red=np.uint8([[[0,0,255]]])
    red2=np.uint8([[[46,5,255]]])

    hsv_b = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    hsv_g = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    hsv_r = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
    hsv_r2 = cv2.cvtColor(red2, cv2.COLOR_BGR2HSV)

    print('hsv_b=', hsv_b)
    print('hsv_g=', hsv_g)
    print('hsv_r=', hsv_r)
    print('hsv_r2=', hsv_r2)

    # 빨간색 범위만 추출
    img = cv2.imread('image/girlgroup06.jpg')
    cv2.imshow('org', img)

    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red_low = np.array([0, 100, 100])
    red_high = np.array([10, 255, 255])
    redmask = cv2.inRange(imghsv, red_low, red_high)

    red_low2 = np.array([170, 100, 100])
    red_high2 = np.array([180, 255, 255])
    redmask2 = cv2.inRange(imghsv, red_low2, red_high2)

    print('redmask=', redmask)
    print('redmask2=', redmask2)

    redmask3 = cv2.bitwise_or(redmask, redmask2)
    print('redmask3 shape=', redmask3.shape)

    redmask4 = cv2.threshold(redmask3, 1, 255, cv2.THRESH_BINARY)
    print('redmask4 shape=', redmask4[1])

    redimg = cv2.bitwise_and(img, img, mask=redmask3)
    # redimg = cv2.bitwise_and(img, img, mask=redmask4[1])
    # redimg2 = cv2.bitwise_and(img, img, mask=redmask2)

    cv2.imshow('redimg', redimg)
    # cv2.imshow('redimg2', redimg2)

    cv2.waitKey(0)


hsv()

__author__ = 'crazyj'

import numpy as np
import cv2
import matplotlib.pyplot as plt


def contourtest():
    img = cv2.imread('image/girl05.png', cv2.IMREAD_GRAYSCALE)
    img_copy = img.copy()

    ret, thr = cv2.threshold(img, 100, 255, 0)
    _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0,0,255), 1)
    cv2.imshow('thresh', thr)
    cv2.imshow('contour', img)


    # 특정 컨투어의 면적, 무게중심 둘레길이 구하기
    idx = 0
    areamax = 0
    contourscnt = len(contours)

    print( ' contours cnt=', contourscnt)
    for i in range(contourscnt):
        c=contours[i]
        area = cv2.contourArea(c)
        if areamax<area:
            idx=i
            areamax=area

    print('max big contour index=', idx)
    contour = contours[idx]

    img_copy=np.zeros_like(img_copy)
    cv2.drawContours(img_copy, [contour], -1, (255, 255, 255), 1)
    area = cv2.contourArea(contour)
    length = cv2.arcLength(contour, True)
    mmt = cv2.moments(contour)
    cx = int( mmt['m10']/mmt['m00'])
    cy = int (mmt['m01']/mmt['m00'])
    print('area=', area)
    print('  m00=', mmt['m00'])
    print('length=', length)
    print('center=', cx, cy)

    cv2.circle(img_copy, (cx, cy), 10, (255,0,0))

    cv2.imshow('area', img_copy)


contourtest()
cv2.waitKey(0)
cv2.destroyAllWindows()



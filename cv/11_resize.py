__author__ = 'crazyj'

import numpy as np
import cv2


# resize
def transform():
    img = cv2.imread('image/girl07.png')
    h, w = img.shape[:2]

    img2 = cv2.resize(img, None, fx=0.5, fy=1, interpolation=cv2.INTER_AREA)
    img3 = cv2.resize(img, None, fx=1, fy=0.5, interpolation=cv2.INTER_AREA)
    img4 = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    cv2.imshow('org', img)
    cv2.imshow('fx=0.5', img2)
    cv2.imshow('fy=0.5', img3)
    cv2.imshow('fxy=0.5', img4)


# transform()


def move():
    img = cv2.imread('image/girl07.png')
    h, w = img.shape[:2]

    M=np.float32( [[1,0, 100], [0,1,50]])       # x 100,  y 50 move

    # img2 = cv2.warpAffine(img, M, (w,h))
    img2 = cv2.warpAffine(img, M, (w+100,h+50))
    cv2.imshow('shift image', img2)

# move()


def rotate():
    # img = cv2.imread('image/girl07.png')
    img = cv2.imread('image/dave.jpg')
    h, w = img.shape[:2]

    M1 = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
    M2 = cv2.getRotationMatrix2D((w/2, h/2), 90, 1)

    img2 = cv2.warpAffine(img, M1, (w,h))
    img3 = cv2.warpAffine(img, M2, (w,h))

    cv2.imshow('45 rotate', img2)
    cv2.imshow('90 rotate', img3)

# rotate()

# 회전 ; 임의 포인트3개 변화 기반
def afftr():
    img = cv2.imread('image/dave.jpg')
    h, w = img.shape[:2]

    # move 3 points to next position.
    pts1 = np.float32( [[50,50], [200,50], [20,200]])
    pts2 = np.float32( [[10,100], [200, 50], [100,250]])

    M = cv2.getAffineTransform(pts1, pts2)

    img2 = cv2.warpAffine(img, M, (w,h))
    cv2.imshow('affine-transform', img2)

# afftr()

# 원근감  ; 임의 포인트 4개
def perspect():
    # img = cv2.imread('image/dave.jpg')
    # img = cv2.imread('image/girl07.png')
    img = cv2.imread('image/rail.png')
    h, w = img.shape[:2]
    # move 4 points to next position.
    pts1 = np.float32( [[0,0], [657,0], [0,556], [657,556]])
    pts2 = np.float32( [[300,65], [350, 52], [28,550], [650, 530]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img2 = cv2.warpPerspective(img, M, (w,h))   # 주의!
    cv2.imshow('affine-transform', img2)

perspect()

cv2.waitKey(0)
cv2.destroyAllWindows()



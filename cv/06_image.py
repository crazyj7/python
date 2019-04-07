import numpy as np
import cv2

img = cv2.imread('image/girlgroup01.jpg')
cv2.imshow('girl', img)
print('img shape=', img.shape)


# 260,10 ~ 340,340
img_crop = img[10:340, 260:340]
cv2.imshow('crop', img_crop)

b,g,r = cv2.split(img)
cv2.imshow('red', r)

img2 = img
img2[:,:,2]=0
cv2.imshow('redoff', img2)

grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('grayimg shape=', grayimg.shape)
cv2.imshow('grayimg', grayimg)
grayimg2col = cv2.cvtColor(grayimg, cv2.COLOR_GRAY2BGR)
print('grayimg2col shape=', grayimg2col.shape)
cv2.imshow('grayimg2col', grayimg2col)

cv2.waitKey(0)
cv2.destroyAllWindows()



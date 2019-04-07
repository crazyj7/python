import numpy as np
import cv2

img = cv2.imread('image/bk01.jpg')
# cv2.imshow('back', img)

print('image ; shape=', img.shape)
img2 = cv2.imread('image/girl09.PNG')

#crop
img_crop = img[0:img2.shape[0], 0:img2.shape[1]]
cv2.imshow('back', img_crop)
cv2.imshow('girl', img2)

# plus?
#merge1 = img_crop+img2
merge1 = cv2.add(img_crop, img2)
cv2.imshow('plus1', merge1)

cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import numpy as np

img = cv2.imread('resource/ship1.png', cv2.IMREAD_UNCHANGED)
img = cv2.resize(img, (100, 100))
cv2.imwrite('resource/ship1_r.png', img)
cv2.imshow('test', img)
cv2.waitKey(0)

exit()

# 이미지 90도 회전

img = cv2.imread('resource/rocket1.png', cv2.IMREAD_UNCHANGED)
img = cv2.transpose(img)
img = cv2.flip(img, 1)
img2 = cv2.resize(img, (100, 50))
cv2.imwrite('resource/rocket2.png', img2)
# h,w,c = img2.shape
# mat = cv2.getRotationMatrix2D((w/2, h/2), -90, 1)
# img3 = cv2.warpAffine(img2, mat, (w, h))
cv2.imshow('test', img2)
cv2.waitKey(0)
exit()




# 이미지 좌우 반전

img = cv2.imread('resource/plane.png', cv2.IMREAD_UNCHANGED)
print(img.shape)
img2 = cv2.flip(img, 1)    # left/right flip
print(img2.shape)
cv2.imwrite('resource/plane2.png', img2)

img2_r = cv2.resize(img2, (80, 50))
cv2.imwrite('resource/plane3.png', img2_r)

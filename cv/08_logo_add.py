"""
로고 이미지를 대상 이미지에 추가한다.

1. 로고 이미지가 들어갈 대상 이미지의 영역을 얻는다. ROI
2. 로고 이미지를 흑백으로 바꾸고, threshold에 따라 배경과 전경을 분리한 마스킹을 생성.
3. 로고 이미지에 마스킹 적용하여 전경만 추출.
4. ROI에 마스킹 적용하여 전경만 추출.
5. 두 이미지를 합성하여 ROI를 갱신한다.
6. 대상 이미지에 ROI 영역을 덮어쓴다.

"""

import numpy as np
import cv2

imglogo = cv2.imread('image/logo01.png')  # noise background... background near to white.
img = cv2.imread('image/girlgroup07.jpg')
if imglogo is None or img is None:
    print('open failed.')
    exit(1)

print('imglogo shape=', imglogo.shape)
print('img shape=', img.shape)

rows, cols, channels = imglogo.shape

imglogo_gray = cv2.cvtColor(imglogo, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray logo', imglogo_gray)


ret, mask = cv2.threshold(imglogo_gray, 240, 255, cv2.THRESH_BINARY)    # white color mask. back.
mask_inv = cv2.bitwise_not(mask)
# cv2.imshow('mask', mask)
# cv2.imshow('mask_inv', mask_inv)
imglogo_fg = cv2.bitwise_and(imglogo, imglogo, mask=mask_inv)
cv2.imshow('logo fg', imglogo_fg)

vpos=10
hpos=10
roi = img[vpos:rows+vpos, hpos:cols+hpos]
roi_bg = cv2.bitwise_and(roi, roi, mask=mask)

roi_update = cv2.add(roi_bg, imglogo_fg)
img[vpos:vpos+rows, hpos:hpos+cols]=roi_update
cv2.imshow('logo+', img)


cv2.waitKey(0)
cv2.destroyAllWindows()



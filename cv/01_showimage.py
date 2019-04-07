import numpy as np
import cv2

def showImage(imgfile):
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)

    # 윈도우 크기 조절 가능
    cv2.namedWindow('test image', cv2.WINDOW_NORMAL)

    cv2.imshow('test image', img)
    k = cv2.waitKey(0)
    if k==27:
        cv2.destroyAllWindows()
    elif k==ord('c'):
        cv2.imwrite('image/copy01.jpg', img)
    elif k==ord('g'):
        img2 = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite('image/gray01.jpg', img2)

# showImage('dave.jpg')
showImage('image/girl01.PNG')



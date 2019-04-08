import numpy as np
import cv2

def onChange(x):
    pass

def Trackbar():
    img = np.zeros((300,400,3), np.uint8)
    cv2.namedWindow('color', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('B', 'color', 0, 255, onChange)
    cv2.createTrackbar('G', 'color', 0, 255, onChange)
    cv2.createTrackbar('R', 'color', 0, 255, onChange)

    switch = '0: OFF\n1: ON'
    cv2.createTrackbar(switch, 'color', 0, 1, onChange)

    while True:
        cv2.imshow('color', img)
        k=cv2.waitKey(1) & 0xFF
        if k==27:
            break
        b=cv2.getTrackbarPos('B', 'color')
        g=cv2.getTrackbarPos('G', 'color')
        r=cv2.getTrackbarPos('R', 'color')
        s=cv2.getTrackbarPos(switch, 'color')

        if s==0:
            img[:]=0
        else:
            img[:] = [b,g,r]

    cv2.destroyAllWindows()

Trackbar()






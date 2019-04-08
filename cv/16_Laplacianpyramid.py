__author__ = 'crazyj'


import numpy as np
import cv2
import matplotlib.pyplot as plt


def pyramid():
    img = cv2.imread('image/girl04.png', cv2.IMREAD_GRAYSCALE)
    tmp = img.copy()

    titles=['org', 'level1', 'level2', 'level3']
    g_down=[]
    g_down.append(tmp)

    for i in range(3):
        tmp1 = cv2.pyrDown(tmp)
        g_down.append(tmp1)
        tmp=tmp1

    # up test
    uptitles=['level1up', 'level2up', 'level3up']
    g_up=[]
    for i in range(3):
        tmp = g_down[i+1]
        tmp1 = cv2.pyrUp(tmp)
        tmp1 = cv2.resize(tmp1, dsize=(g_down[i].shape[1], g_down[i].shape[0]), interpolation=cv2.INTER_CUBIC)

        g_up.append(tmp1)

    for i in range(3):
        tmp = cv2.subtract(g_down[i], g_up[i])
        cv2.imshow(uptitles[i], tmp)

pyramid()
cv2.waitKey(0)
cv2.destroyAllWindows()


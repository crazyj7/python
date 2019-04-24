__author__ = 'crazyj'

import numpy as np
import cv2
import matplotlib.pyplot as plt


def imginfo():

    # draw random polygon
    count = 15
    randompts = np.random.randint(0, 500, (count,2))
    print(randompts)
    # img = np.zeros((500,500), np.uint8)
    img = np.zeros((500,500, 3), np.uint8)
    for i in range(count-1):
        # cv2.line(img, tuple(randompts[i]), tuple(randompts[i+1]), 255, 3)
        cv2.line(img, tuple(randompts[i]), tuple(randompts[i + 1]), (255,255,255), 3)
    cv2.imshow('org image', img)

    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # get image info...
    # aspect ratio
    x,y,w,h = cv2.boundingRect(imggray)
    aspect_ratio = float(w)/h
    print('aspect_ratio=', aspect_ratio, 'w=', w, 'h=', h)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,200), 1)     # red ; boundingRect area

    # extent
    contours,_ = cv2.findContours(imggray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt=contours[0]

    cv2.drawContours(img, [cnt], 0, (0,255,0), 2)   # green ; contour area

    area = cv2.contourArea(cnt)
    rect_area = w*h
    extent = float(area) / rect_area
    print('extent=', extent, 'contourArea=', area, 'rectarea=', rect_area)

    #solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area
    print('solidity=', solidity, 'contourArea=', area, 'hull_area=',hull_area)
    cv2.drawContours(img, [hull], 0, (255,0,0), 1)  # blue ; convex hull area

    # moment
    mmt = cv2.moments(cnt)
    # weight center
    cx = int( mmt['m10']/mmt['m00'])
    cy = int( mmt['m01']/mmt['m00'])
    print('moment center=', cx, cy)
    cv2.circle(img, (cx,cy), 3, (0,0,255), -1)  # moment center.

    # equivalent diameter
    eq_dia = np.sqrt(4*area/np.pi)
    print('equivalent diameter=', eq_dia)
    eq_radius = int(eq_dia/2)
    cv2.circle(img, (cx,cy), eq_radius, (0,0,255), 2)   # red circle

    # orientation
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(img, ellipse, (50,50,50), 2)    # fit ellipse
    print('orientation=', ellipse[2])


    # extream points
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    print('leftmost=', leftmost, 'rightmost=', rightmost, 'topmost=', topmost, 'bottommost=', bottommost)
    cv2.circle(img, leftmost, 3, (0,0,255), -1)  # moment center.
    cv2.circle(img, rightmost, 3, (0,0,255), -1)  # moment center.
    cv2.circle(img, topmost, 3, (0,0,255), -1)  # moment center.
    cv2.circle(img, bottommost, 3, (0,0,255), -1)  # moment center.

    cv2.imshow('info', img)



imginfo()
cv2.waitKey(0)
cv2.destroyAllWindows()



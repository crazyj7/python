import numpy as np
import cv2

def Drawing():
    img = np.zeros((320,480,3), np.uint8)   # 480x320x3(channel)

    cv2.line(img, (10,10), (470,310), (255,0,0), 5)     # 대각선으로 그은 선. 두께는 5, Blue색. BGR 순서임.
    cv2.rectangle(img, (100,100), (200,200), (0,0,255), 2)  # 빨간색 박스
    cv2.circle(img, (200,200), 50, (0, 255,0), 2)
    cv2.ellipse(img, (300, 200), (100, 50), 0, 0, 180, (255,255,0), -1) # 타원중심 점,  x반경100, y반경50 크기.
            # thick을 -1로 주면 색을채움.

    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Hello CV2', (10,100), font, 3, (0,0,255), 2)

    cv2.imshow('drawing', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

Drawing()



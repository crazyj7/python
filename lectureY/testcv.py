
import numpy as np
import cv2


def showImage(imgfile):
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)        # 이미지 파일 로딩
    print('image shape=', img.shape)
    img = cv2.resize(img, (960, 540))  # resize. too big
    print('image shape=', img.shape)
    cv2.imshow('title sample name', img)                # 이미지 출력
    cv2.waitKey(0)                    # 아무키나 기다림. delay 단위는 millisec. 0이면 계속 입력을 기다린다.
    cv2.destroyAllWindows()


showImage('snow.jpg')


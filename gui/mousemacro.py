import pyautogui as pag
import mss, sys
import cv2
import numpy as np

'''
윈도우에서 pyautogui 설치 에러. cp949 error시 버전 지정.
pip install pyautogui==0.9.38
또는  아래 두 커맨드
pip install pygetwindow==0.0.1
pip install pyautogui

'''

left_icon_pos = {'left':10, 'top':10, 'width':100, 'height':200}

pag.PAUSE = 0.01  #클릭 후 대기 시간 딜레이.


def click(coords):
    pag.moveTo(x=coords[0], y=coords[1], duration=0.0)
    pag.mouseDown()
    pag.mouseUp()

# 해당 위치 이미지 캡쳐 -> 평균값 r,g,b가 각각 임계 범위이면 해당 타입으로 구분.

with mss.mss() as sct:

    # 화면 캡쳐
    left_img = np.array(sct.grab(left_icon_pos))[:,:,:3]
    right_img = np.array(sct.grab({'left':100, 'top':100,
                                   'width':200, 'height':200}))[:,:,:3]
    print(left_img)
    cv2.imshow('leftimg', left_img)
    cv2.imshow('rightimg', right_img)
    cv2.waitKey(0)
sys.exit()

while True:
    # 마우스 위치 get.
    x,y = pag.position()
    msg = str(x)+','+str(y)
    print(msg)

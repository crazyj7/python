'''
문서를 찍은 이미지를
가로 세로 사각형으로 변형시켜 스캔본처럼 만든다.

문서 사각 영역을 left,top  click   right, top click
right, bottom click,   left, bottom click 이런식으로
시계방향으로 돌아가면서 꼭지점을 클릭하면 사각형을 평평하게 펴준다.
'''

import cv2, os
import numpy as np

img_path = 'dave.jpg'
filename, ext = os.path.splitext(img_path)

ori_img = cv2.imread(img_path)

src = []

# mouse eventhandler
def mouse_handler(event, x, y, flags, param):
    if (event==cv2.EVENT_LBUTTONUP) :
        img = ori_img.copy()
        src.append([x,y])

        for xx, yy in src:
            cv2.circle(img, center=(xx,yy), radius=5, color=(0,255,0),
                       thickness=1, lineType=cv2.LINE_AA)
        cv2.imshow('img', img)

        if len(src)==4:
            src_np = np.array(src, dtype=np.float32)
            width = max(np.linalg.norm(src_np[0]-src_np[1]), np.linalg.norm(src_np[2]-src_np[3]))
            height = min(np.linalg.norm(src_np[0]-src_np[3]), np.linalg.norm(src_np[1]-src_np[2]))

            dst_np = np.array( [[0,0], [width,0], [width, height], [0, height]], dtype=np.float32)

            M = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)
            result = cv2.warpPerspective(ori_img, M=M, dsize=(width, height))

            cv2.imshow('result', result)
            cv2.imwrite(filename+'_output.'+ext, result)

cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse_handler)

cv2.imshow('img', ori_img)
cv2.waitKey(0)


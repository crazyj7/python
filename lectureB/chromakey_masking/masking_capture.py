'''

마스킹하여 캡쳐하기 update

마스킹 색상 선택.  ; 투명망토가 보이는 상태로 스페이스키 누르고, 마우스 드래그로 영역 선택. 스페이스키.
배경 화면 선택. ; 배경만 나오게 하고, 스페이스키.
지정된 색 범위를 마스킹 영역으로 사용하여 영상 투명 필터링. ; 이제 투명망토를 써보자.

'''


import cv2
import numpy as np
import time, argparse



def onChange(x):
    pass



# 투명 컬러로 사용할 값을 화면에서 선택한다.
cap = cv2.VideoCapture(0)
if not cap.isOpened() :
    exit()
while True:
    ret, bg1 = cap.read()
    cv2.imshow('select color', bg1)
    key = cv2.waitKey(1)
    if key==32 or key==27:
        break
cap.release()
rect = cv2.selectROI('select color', bg1, fromCenter=False, showCrosshair=True)
# 마우스로 드래그하여 선택하여 스페이스키로 완료.
cv2.destroyWindow('select color')
print(rect)

maskpart = bg1[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
maskpart = cv2.cvtColor(maskpart, cv2.COLOR_BGR2HSV)        # HSV: 색조, 선명도, 밝기

print(maskpart.shape, type(maskpart[0,0]))
maskpart = maskpart.astype(np.int)
maskpart = np.reshape(maskpart, (-1, 3))
print(maskpart.shape)

# H ; 0~180  ,  S,V : 0~255
# 문제는 빨간색은 H가 0=180이다. 따라서 원형으로 배치되는 것으로 바운더리 체크가
# 달라져야 한다. overflow, underflow 계산이 필요함.
# type이 0~255 범위라서 overflow되면 자동으로 넘어간다. 이를 일단 방지하고 수동처리.
# 따라서 type 변경.

def colorRange(maskpart, h, s, v):
    mmin = np.min(maskpart, axis=0)
    mmax = np.max(maskpart, axis=0)
    mmin[0] -= h
    mmax[0] += h
    mmin[1] -= s
    mmax[1] += s
    mmin[2] -= v
    mmax[2] += v
    return mmin, mmax

h=20
s=v=50
mmin, mmax = colorRange(maskpart, h, s, v)
print('min=', mmin)
print('max=', mmax)
# exit(0)


parser = argparse.ArgumentParser()
parser.add_argument('--video', help='Input video path')
args = parser.parse_args()

cap = cv2.VideoCapture(args.video if args.video else 0)

time.sleep(0.1)

# Grap background image from first part of the video
while True:
    ret, background = cap.read()
    cv2.imshow('press space', background)
    if cv2.waitKey(1)==32:
        break
cv2.destroyWindow('press space')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('videos/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (background.shape[1], background.shape[0]))
out2 = cv2.VideoWriter('videos/original.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                       (background.shape[1], background.shape[0]))


# track bar window
cv2.namedWindow('color', cv2.WINDOW_NORMAL)
cv2.createTrackbar('H', 'color', 0, 100, onChange)
cv2.createTrackbar('S', 'color', 0, 100, onChange)
cv2.createTrackbar('V', 'color', 0, 100, onChange)
cv2.setTrackbarPos('H', 'color', h)
cv2.setTrackbarPos('S', 'color', s)
cv2.setTrackbarPos('V', 'color', v)

while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break

    h=cv2.getTrackbarPos('H', 'color')
    s=cv2.getTrackbarPos('S', 'color')
    v=cv2.getTrackbarPos('V', 'color')

    # Convert the color space from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generate mask to detect red color
    # lower_red = np.array([0, 120, 70])
    # upper_red = np.array([10, 255, 255])
    # mask1 = cv2.inRange(hsv, lower_red, upper_red)
    #
    # lower_red = np.array([170, 120, 70])
    # upper_red = np.array([180, 255, 255])
    # mask2 = cv2.inRange(hsv, lower_red, upper_red)
    #
    # mask1 = mask1 + mask2

    mmin, mmax = colorRange(maskpart, h, s, v)
    lower = np.array([mmin[0], mmin[1], mmin[2] ])
    upper = np.array([mmax[0], mmax[1], mmax[2] ])

    # H ; 0 case. (red color)
    if mmin[0]>0 and mmax[0]<180 :
        mask1 = cv2.inRange(hsv, lower, upper)
    else:
        # print('special case color')
        # special case. red color
        hlist = maskpart[:,0]
        if np.sum(hlist<90)>0 :
            high = np.max(hlist[hlist<90])
        else:
            high=0
        if np.sum(hlist>90)>0 :
            low = np.min(hlist[hlist>90])
        else:
            low=180
        high = high+h
        low = low-h
        # print(low, high)
        # print(mmin, mmax)
        lower = np.array([0, mmin[1], mmin[2]])
        upper = np.array([high, mmax[1], mmax[2]])
        mask1 = cv2.inRange(hsv, lower, upper)
        lower = np.array([low, mmin[1], mmin[2]])
        upper = np.array([180, mmax[1], mmax[2]])
        mask2 = cv2.inRange(hsv, lower, upper)
        mask1 = cv2.bitwise_or(mask1, mask2)

    # lower_black = np.array([0, 0, 0])
    # upper_black = np.array([255, 255, 80])
    # mask1 = cv2.inRange(hsv, lower_black, upper_black)

    '''
    # Refining the mask corresponding to the detected red color
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    '''
    # Remove noise
    mask_cloak = cv2.morphologyEx(mask1, op=cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8), iterations=2)
    mask_cloak = cv2.dilate(mask_cloak, kernel=np.ones((3, 3), np.uint8), iterations=1)
    mask_bg = cv2.bitwise_not(mask_cloak)

    cv2.imshow('mask_cloak', mask_cloak)

    # Generate the final output
    res1 = cv2.bitwise_and(background, background, mask=mask_cloak)
    res2 = cv2.bitwise_and(img, img, mask=mask_bg)
    result = cv2.addWeighted(src1=res1, alpha=1, src2=res2, beta=1, gamma=0)

    cv2.imshow('res1', res1)

    # cv2.imshow('ori', img)
    cv2.imshow('result', result)
    out.write(result)
    out2.write(img)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
out2.release()
cap.release()
cv2.destroyAllWindows()


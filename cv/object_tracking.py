import cv2
import numpy as np


'''
동영상 첫 프레임을 열고 이미지를 띄운다
마우스로 추적 영역을 드래그 선택한 후 , 스페이스키로 완료
동영상이 재생되면서 오브젝트를 추적한다.


cv2의 트랙커 사용.

'''


video_path = 'c:/users/crazy/downloads/bts3.mkv'

# output_size = (375, 667)  # w, h
output_size = (220, 330)  # w,h

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    exit()

# initialize tracker
OPENCV_OBJECT_TRACKERS = {
  "csrt": cv2.TrackerCSRT_create,
  "kcf": cv2.TrackerKCF_create,
  "boosting": cv2.TrackerBoosting_create,
  "mil": cv2.TrackerMIL_create,
  "tld": cv2.TrackerTLD_create,
  "medianflow": cv2.TrackerMedianFlow_create,
  "mosse": cv2.TrackerMOSSE_create
}
tracker = OPENCV_OBJECT_TRACKERS['csrt']()

# read first frame
ret, img = cap.read()
cv2.namedWindow('Select Window')
cv2.imshow('Select Window', img)

fit_to = 'height'
# initialize writing video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)

# set ROI
rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
# 마우스로 드래그하여 선택하여 스페이스키로 완료.
cv2.destroyWindow('Select Window')
tracker.init(img, rect)


while True:
    ret, img = cap.read()
    if not ret:
        break

    success, box = tracker.update(img)
    left, top, w, h = [int(v) for v in box]
    center_x, center_y = left+w/2, top+h/2

    result_top = int(center_y - output_size[1]/2)
    result_bottom = int(center_y + output_size[1]/2)
    result_left = int(center_x - output_size[0]/2)
    result_right = int(center_x + output_size[0]/2)

    result_img = img[result_top:result_bottom, result_left:result_right]
    if result_img.shape[0]>0 and result_img.shape[1]>0 :
        cv2.imshow('out', result_img)
        out.write(result_img)       # write output mpg
        cv2.rectangle(img, pt1=(left,top), pt2=(left+w, top+h),
                      color=(255,255,255), thickness=2)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break


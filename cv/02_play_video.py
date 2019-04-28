import numpy as np
import cv2

def playVideo(vfile):
    try:
        cap = cv2.VideoCapture(vfile)   # 웹캠에서 읽으려면 장치번호 0을 입력.
    except:
        print('capture open failed.')
        return

    #캠 크기
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    # cap.set(3, 480)
    # cap.set(4, 320)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('read failed.')
            break
        cv2.imshow('video', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print('ESC key pressed for terminating.')
            break
    cap.release()
    cv2.destroyAllWindows()


playVideo(0)    # from webcam
# playVideo('image/a.avi')


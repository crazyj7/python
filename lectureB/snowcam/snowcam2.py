import numpy as np
import cv2, sys
import dlib

'''
PYTHON 3.6
sample face video ; 
videos.pexels.com/search/face

동영상에 얼굴 영역을 감지하여 다른 이미지를 덮어쓴다.

cv DNN을 적용해 봤다. 학습데이터는 renet.

'''


# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    try:
        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)
        bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = cv2.add(img1_bg, img2_fg)
    except:
        # ignore
        print('exception in overlay!')
        bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = 0

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img




scaler = 1.0
cap = cv2.VideoCapture(0)

# scaler = 0.3
# cap = cv2.VideoCapture('Slow Motion Video Of People.mp4')

overlay = cv2.imread('ryan_transparent.png', cv2.IMREAD_UNCHANGED)

# detector = dlib.get_frontal_face_detector()
# print('detector=', detector)
# predictor = dlib.shape_predictor('../MODELS/shape_predictor_68_face_landmarks.dat')
# print('predictor=', predictor)

net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 이미지 좌우 반전.
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (int(frame.shape[1]*scaler), int(frame.shape[0]*scaler)))
    result = frame.copy()

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        boxcolor=(0,0,255)
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        if confidence > 0.5:    # 0.4 이상만 이미지 출력.
            boxcolor=(0,255,0)
            face_size = np.max([endX-startX, endY-startY])
            center_x = np.mean([startX, endX])
            center_y = np.mean([startY, endY])
            result = overlay_transparent(result, overlay, center_x, center_y-20, overlay_size=(face_size, face_size))

        if confidence>0.4:  # 0.3 이상만 박스 그림.
            # 박스를 전부 그리면 너무 많다.
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          boxcolor, 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, boxcolor, 2)

    cv2.imshow('img2', result)
    cv2.imshow('img', frame)
    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)





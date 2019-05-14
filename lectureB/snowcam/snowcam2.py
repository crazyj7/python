import numpy as np
import cv2, sys

'''
PYTHON 3.6
sample face video ; 
videos.pexels.com/search/face

동영상에 얼굴 영역을 감지하여 다른 이미지를 덮어쓴다.

dlib은 정면에 모두 보여야 잘 잡힌다. 일부에서 잘 작동하지 않아서
 cv DNN을 적용해 봤다. 학습데이터는 renet. 
'''




'''
safe_copy ; numpy array 안전하게 복사.
목적지, 목적지내의 오프셋 지점. 
src ; 복사할 원본.
오프셋이 음수일때 대응? 보이는 부분만 복사. (오프셋이 양수인 영역)
src의 길이가 dst의 오프셋 지점에 넣으면 overflow 발생시 ; 안전한 처리. valid area.
'''
def safe_copy(dst, startx, starty, src):
    dh, dw = dst.shape[:2]
    sh, sw = src.shape[:2]
    dstx = startx
    dsty = starty
    srcx = 0
    srcy = 0
    if dstx<0 :
        dstx = 0
        srcx = abs(dstx)
    if dsty<0 :
        dsty = 0
        srcy = abs(dsty)
    srcx2 = sw
    srcy2 = sh
    if dstx+sw>dw:
        srcx2 -= (dstx+sw)-dw
    if dsty+sh>dh:
        srcy2 -= (dsty+sh)-dh
    cw = srcx2 - srcx
    ch = srcy2 - srcy
    dst[dsty:dsty+ch,dstx:dstx+cw] = src[srcy:srcy+ch, srcx:srcx+cw]






# overlay function
'''
대상이미지, 추가할 레이어. 추가할 위치 중앙 지점, 영영 크기. 레이어를 늘릴 것인지 여부.
'''
def overlay_transparent(background_img, img_to_overlay_t, center_x, center_y, facew, faceh, bstretch=True):
    bg_img = background_img.copy()

    facex = center_x - int(facew/2)
    facey = center_y - int(faceh/2)

    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if bstretch:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), (facew, faceh))
    else:
        # 레이어 크기 비율 유지.
        s1, s2, _ = img_to_overlay_t.shape
        oo = max(facew / s1, faceh / s2)        ## min or max
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), ( int(s1*oo), int(s2*oo) ) )
        # print('oo=', oo, img_to_overlay_t.shape)

    # 레이어 크기.
    h, w, _ = img_to_overlay_t.shape
    if bstretch==False:
        facex = facex + int((facew - w) / 2)
        facey = facey + int((faceh - h) / 2)
    if facex<0:
        facex=0
    if facey<0:
        facey=0
    if facex+w > bg_img.shape[1] :
        w -= bg_img.shape[1] - (facex+w)
    if facey+h > bg_img.shape[0] :
        h -= bg_img.shape[0] - (facey+h)

    roi_endy = min (facey+h, bg_img.shape[0])
    roi_endx = min (facex+w, bg_img.shape[1])
    roi = bg_img[facey: roi_endy, facex: roi_endx]
    # print(facey, roi_endy, facex, roi_endx, roi.shape)
    # roi2 = roi.copy()
    img_to_overlay_t = img_to_overlay_t[0:roi.shape[0], 0:roi.shape[1]]
    # print(img_to_overlay_t.shape, roi2.shape)
    b, g, r, a = cv2.split(img_to_overlay_t)
    mask = cv2.medianBlur(a, 5)

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)
    # bg_img[facey: facey + h, facex:facex + w] = cv2.add(img1_bg, img2_fg)
    safe_copy(bg_img, facex, facey, cv2.add(img1_bg, img2_fg))

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img




# 화면 크기 조정.
scaler = 1.0

# 얼굴 영역 크기 조정
facescale = 1.0
cap = cv2.VideoCapture(0)

# scaler = 0.3
# cap = cv2.VideoCapture('Slow Motion Video Of People.mp4')

overlay = cv2.imread('ryan_transparent.png', cv2.IMREAD_UNCHANGED)
# overlay = cv2.imread('head6.png', cv2.IMREAD_UNCHANGED)
if overlay.shape[2] == 3:
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

print('overlay shape=', overlay.shape)

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
        if confidence > 0.4:    # 0.4 이상만 이미지 출력.
            boxcolor=(0,255,0)
            face_size = np.max([endX-startX, endY-startY])
            center_x = int(np.mean([startX, endX]))
            center_y = int(np.mean([startY, endY]))
            result = overlay_transparent(result, overlay, center_x, center_y, int(facescale*(endX-startX)), int(facescale*(endY-startY)), bstretch=False)

        if confidence > 0.3:  # 0.3 이상만 박스 그림.
            center_x = int(np.mean([startX, endX]))
            center_y = int(np.mean([startY, endY]))
            # 박스를 전부 그리면 너무 많다.
            fw = int (facescale*(endX-startX))
            fh = int (facescale*(endY-startY))
            startX = center_x-int(fw/2)
            startY = center_y-int(fh/2)
            endX = center_x+int(fw/2)
            endY = center_y+int(fh/2)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          boxcolor, 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, boxcolor, 2)

    cv2.imshow('img2', result)
    cv2.imshow('img', frame)
    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)





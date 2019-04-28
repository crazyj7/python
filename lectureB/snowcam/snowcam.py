import numpy as np
import cv2, sys
import dlib

'''
PYTHON 3.6
sample face video ; 
videos.pexels.com/search/face

동영상에 얼굴 영역을 감지하여 다른 이미지를 덮어쓴다.

dlib의 얼굴 감지 모듈 ; dlib.get_frontal_face_detector()
학습 파일이 필요 ; predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


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

detector = dlib.get_frontal_face_detector()
print('detector=', detector)
predictor = dlib.shape_predictor('../MODELS/shape_predictor_68_face_landmarks.dat')
print('predictor=', predictor)

while True:
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.resize(img, (int(img.shape[1]*scaler), int(img.shape[0]*scaler)))
    ori = img.copy()
    result = ori

    faces = detector(img)
    # print('faces=', faces)
    if faces!=None and len(faces)>0 :
        # face = faces[0]
        for face in faces:
            dlib_shape = predictor(img, face)
            shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

            # compute center of face
            top_left = np.min(shape_2d, axis=0)
            bottom_right = np.max(shape_2d, axis=0)
            center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)
            face_size = int( max(bottom_right-top_left) * 1.2)

            result = overlay_transparent(ori, overlay, center_x, center_y-20, overlay_size=(face_size, face_size))
            ori = result    # overwrite

            img = cv2.rectangle(img, pt1=(face.left(), face.top()),
                    pt2=(face.right(), face.bottom()), color=(255,255,255),
                    thickness=2, lineType=cv2.LINE_AA)
            for s in shape_2d:
                cv2.circle(img, center=tuple(s), radius=1, color=(255,255,255),
                           thickness=2, lineType=cv2.LINE_AA)
            # cv2.circle(img, center=tuple(top_left), radius=1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)
            # cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('img2', result)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)





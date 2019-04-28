'''
dlib face detector

testing...

'''

import numpy as np
import cv2, sys
import dlib



scaler = 1.0
cap = cv2.VideoCapture(0)

# scaler = 0.3
# cap = cv2.VideoCapture('Slow Motion Video Of People.mp4')

detector = dlib.get_frontal_face_detector()
print('detector=', detector)
predictor = dlib.shape_predictor('../MODELS/shape_predictor_68_face_landmarks.dat')
print('predictor=', predictor)

while True:
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.resize(img, (int(img.shape[1]*scaler), int(img.shape[0]*scaler)))
    faces = detector(img)
    print('faces count=', len(faces))
    if faces!=None and len(faces)>0 :
        for face in faces:
            dlib_shape = predictor(img, face)
            shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

            # compute center of face
            top_left = np.min(shape_2d, axis=0)
            bottom_right = np.max(shape_2d, axis=0)
            center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)
            face_size = int( max(bottom_right-top_left) * 1.2)

            img = cv2.rectangle(img, pt1=(face.left(), face.top()),
                    pt2=(face.right(), face.bottom()), color=(255,255,255),
                    thickness=2, lineType=cv2.LINE_AA)
            for s in shape_2d:
                cv2.circle(img, center=tuple(s), radius=1, color=(255,255,255),
                           thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)



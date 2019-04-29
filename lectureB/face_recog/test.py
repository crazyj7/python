'''
캠에 나온 사람과 지정한 이미지 사진의 거리 측정.
'''

import dlib, cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('../models/dlib_face_recognition_resnet_model_v1.dat')

def read_img(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def encode_face(img):
  dets = detector(img, 1)
  if len(dets) == 0:
    return np.empty(0)
  for k, d in enumerate(dets):
    # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    return np.array(face_descriptor)

# main
# img1_path = 'you2.jpg' # iu
# img2_path = '/Users/visualcamp/Pictures/00502318_20180518.JPG' # suz
# img2_path = '/Users/visualcamp/Pictures/660190_v9_ba.jpg' # suz
# img2_path = '/Users/visualcamp/Development/tf/GazeCapture/dataset/processed/00002/frames/00000.jpg'
# img1_path = '/Users/visualcamp/Development/tf/GazeCapture/dataset/processed/03523/frames/02190.jpg'
# img2_path = '/Users/visualcamp/Development/tf/GazeCapture/dataset/processed/03523/frames/00000.jpg'
# img2_path = '/Users/visualcamp/Development/tf/GazeCapture/dataset/processed/02534/frames/00005.jpg'

img1_path = 'you.jpg'
img1 = read_img(img1_path)
img1_encoded = encode_face(img1)
# img2 = read_img(img2_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
  exit()
while True:
  ret, img2_org = cap.read()
  if not ret:
    break

  img2_org = cv2.resize(img2_org, (640, img2_org.shape[0] * 640 // img2_org.shape[1]))
  img2 = cv2.cvtColor(img2_org, cv2.COLOR_BGR2RGB)
  if cv2.waitKey(1)==ord('q'):
    break

  # img2_encoded = encode_face(img2)
  # cv2.imshow('window', img2_org)
  # if len(img2_encoded) == 0:
  #   continue
  # dist = np.linalg.norm(img1_encoded - img2_encoded, axis=0)
  # print('%s, Distance: %s' % (dist < 0.6, dist))

  dets = detector(img2, 1)
  if len(dets) > 0:
    for k, d in enumerate(dets):
      # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
      shape = sp(img2, d)
      face_descriptor = facerec.compute_face_descriptor(img2, shape)
      dist = np.linalg.norm(img1_encoded - np.array(face_descriptor), axis=0)
      if dist<0.5:
        color = (0,255,0)
        thick = 2
      else:
        color = (255,0,0)
        thick = 1
      cv2.putText(img2_org, "%4.2f"%dist, (d.left(), d.top()), cv2.FONT_ITALIC, 0.5, color, 1, cv2.LINE_AA)
      img2_org = cv2.rectangle(img2_org, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=color,
                               thickness=thick)
  cv2.imshow('cam', img2_org)



'''
face detection and recognition



+models files
https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
https://github.com/kairess/simple_face_recognition/raw/master/models/dlib_face_recognition_resnet_model_v1.dat

'''

import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('../MODELS/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1("../MODELS/dlib_face_recognition_resnet_model_v1.dat")


def read_img(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img


def find_faces(img):
    dets = detector(img, 1)     # 이미지에서 얼굴박스 전부 추출.
    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))   # 얼굴박스
        rects.append(rect)
        shape = sp(img, d)  # 이미지와 얼굴박스 정보 한 개를 주면 얼굴모양 정보를 얻는다.
        # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)    # 얼굴모양 68개 점 기록.
        shapes.append(shape)

    return rects, shapes, shapes_np



def encode_faces(img, shapes):
    face_decriptors = []
    for shape in shapes:
        face_decriptor = facerec.compute_face_descriptor(img, shape)
        face_decriptors.append(np.array(face_decriptor))
    return np.array(face_decriptors)

img_paths = {
    'neo': 'img/neo.jpg',
    'trinity': 'img/trinity.jpg',
    'morpheus': 'img/morpheus.jpg',
    'smith': 'img/smith.jpg'
}
descs = {
    'neo': None,
    'trinity': None,
    'morpheus': None,
    'smith': None
}

if True:
    for name, img_path in img_paths.items():
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        _, img_shapes, _ = find_faces(img_rgb)
        descs[name] = encode_faces(img_rgb, img_shapes)[0]

    np.save('img/descs.npy', descs)
    print(descs)
else:
    descs=np.load('img/descs.npy')



# img_rgb = read_img('img/matrix.jpg')
img_rgb = read_img('you.jpg')

rects, shapes, _ = find_faces(img_rgb)
descriptors = encode_faces(img_rgb, shapes)

fig, ax = plt.subplots(1, figsize=(20, 20))
ax.imshow(img_rgb)

for i, desc in enumerate(descriptors):

    found = False
    for name, saved_desc in descs.items():
        dist = np.linalg.norm([desc] - saved_desc, axis=1)

        if dist < 0.6:
            found = True

            text = ax.text(rects[i][0][0], rects[i][0][1], name,
                           color='b', fontsize=40, fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'), path_effects.Normal()])
            rect = patches.Rectangle(rects[i][0],
                                     rects[i][1][1] - rects[i][0][1],
                                     rects[i][1][0] - rects[i][0][0],
                                     linewidth=2, edgecolor='w', facecolor='none')
            ax.add_patch(rect)
            break

    if not found:
        ax.text(rects[i][0][0], rects[i][0][1], 'unknown',
                color='r', fontsize=20, fontweight='bold')
        rect = patches.Rectangle(rects[i][0],
                                 rects[i][1][1] - rects[i][0][1],
                                 rects[i][1][0] - rects[i][0][0],
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.axis('off')
# plt.savefig('result/output.png')
plt.show()





import random
import dlib, cv2, os
import pandas as pd
import numpy as np

img_size = 224
# dirname = 'CAT_00'

dirnames=['CAT_00', 'CAT_01', 'CAT_02', 'CAT_03', 'CAT_04', 'CAT_05', 'CAT_06']

for dirname in dirnames:
    base_path = './cats/%s' % dirname
    file_list = sorted(os.listdir(base_path))
    random.shuffle(file_list)

    # 원본이미지, 랜드마크, 박스
    dataset = {
      'img': [],
      'lmk': [],
      'bbs': []
    }

    # 수정이미지(박스), 랜드마크
    dataset2 = {
        'img2':[],
        'lmk2':[]
    }

    def resize_img(im):
        old_size = im.shape[:2]  # old_size is in (height, width) format
        ratio = float(img_size) / max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        # new_size should be in (width, height) format
        im = cv2.resize(im, (new_size[1], new_size[0]))
        delta_w = img_size - new_size[1]
        delta_h = img_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                    value=[0, 0, 0])
        return new_im, ratio, top, left

    for f in file_list:
        if '.cat' not in f:
            continue

        print('process:', os.path.join(base_path, f))
        # read landmarks
        pd_frame = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)
        landmarks = (pd_frame.values[0][1:-1]).reshape((-1, 2))

        # load image
        img_filename, ext = os.path.splitext(f)
        img = cv2.imread(os.path.join(base_path, img_filename))     #  big image original
        # cv2.imshow('orgimg', img)

        # resize image and relocate landmarks
        img, ratio, top, left = resize_img(img)         # resize img 224x224 margin black fill. small image
        landmarks = ((landmarks * ratio) + np.array([left, top])).astype(np.int)    # pos in resizing img
        bb = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])       # box pos . no margin.. facefit only

        # compute lazy bounding box for detecting landmarks
        center = np.mean(bb, axis=0)
        face_size = max(np.abs(bb[1] - bb[0]))
        new_bb = np.array([
            center - face_size * 0.6,
            center + face_size * 0.6
        ]).astype(np.int)
        new_bb = np.clip(new_bb, 0, 99999)          ## new box pos. margin include.

        face_img = img[new_bb[0][1]:new_bb[1][1], new_bb[0][0]:new_bb[1][0]]        ## cut image. face + margin img. ; face image
        new_landmarks = landmarks - np.array([new_bb[0][0], new_bb[0][1]])
        face_img, face_ratio, face_top, face_left = resize_img(face_img)            # resize face img 224x224 margin black fill
        new_landmarks = ((new_landmarks* face_ratio) + np.array([face_left, face_top])).astype(np.int)

        # debug view
        # for l in landmarks:
        #     cv2.circle(img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)
        # cv2.imshow('img', img)
        #
        # for l in new_landmarks:
        #     cv2.circle(face_img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)
        # cv2.imshow('faceimg', face_img)
        #
        # if cv2.waitKey(0)==ord('q'):
        #     exit()

        dataset['img'].append(img)
        dataset['lmk'].append(landmarks.flatten())
        dataset['bbs'].append(bb.flatten())

        dataset2['lmk2'].append(new_landmarks.flatten())
        dataset2['img2'].append(face_img)

        # Preview image. mark circle.

        # for l in landmarks:
        #     cv2.circle(img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)
        # cv2.imshow('img', img)
        # if cv2.waitKey(0) == ord('q'):
        #     break

    savefile ='./cats/%s_1.npy' % dirname
    np.save(savefile, np.array(dataset))
    print('savefile=', savefile)

    savefile ='./cats/%s_2.npy' % dirname
    np.save(savefile, np.array(dataset2))
    print('savefile=', savefile)

print('end')

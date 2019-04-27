'''
cat hipster test update
+,
'''

import keras, sys, cv2, os
from keras.models import Model, load_model
import numpy as np
import pandas as pd
from math import atan2, degrees
import random


modelsavefile1 = 'cat_hipster1.model'
modelsavefile2 = 'cat_hipster2.model'

scale=0.5   # 출력 화면 크기
img_size = 224
base_path = './cats/CAT_05'

# file_list = sorted(os.listdir(base_path))
file_list = os.listdir(base_path)
random.shuffle(file_list)

print(file_list)

# this is most important thing
glasses = cv2.imread('./cats/glasses.png', cv2.IMREAD_UNCHANGED)

bbs_model = load_model(modelsavefile1)

lmk_model = load_model(modelsavefile2)


def resize_img(im):
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
    return new_im, ratio, top, left


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
    roi = bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img


def angle_between(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff))


# testing
count = len(file_list)
fileindex=0
while fileindex<count:
    f=file_list[fileindex]
    if '.jpg.cat' in f:
        fileindex+=1
        continue
    if '.jpg' not in f:
        fileindex+=1
        continue
    filename = os.path.join(base_path, f)
    img = cv2.imread(filename)
    img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)) ,
                     interpolation = cv2.INTER_AREA)
    print('filename=', filename, ' scale=', scale)
    if img.any()==None:
        fileindex+=1
        continue

    ori_img = img.copy()
    result_img = img.copy()

    # predict bounding box
    img, ratio, top, left = resize_img(img)

    inputs = (img.astype('float32') / 255).reshape((1, img_size, img_size, 3))
    pred_bb = bbs_model.predict(inputs)[0].reshape((-1, 2))

    # compute bounding box of original image
    ori_bb = ((pred_bb - np.array([left, top])) / ratio).astype(np.int)

    # compute lazy bounding box for detecting landmarks
    center = np.mean(ori_bb, axis=0)
    face_size = max(np.abs(ori_bb[1] - ori_bb[0]))
    new_bb = np.array([
        center - face_size * 0.6,
        center + face_size * 0.6
    ]).astype(np.int)
    new_bb = np.clip(new_bb, 0, 99999)

    face_img = ori_img[new_bb[0][1]:new_bb[1][1], new_bb[0][0]:new_bb[1][0]]
    face_img, face_ratio, face_top, face_left = resize_img(face_img)

    # debug
    # cv2.imshow('org img', ori_img)
    # cv2.imshow('predict face part', face_img)
    # print('press any key....(q is quit)')
    # if cv2.waitKey(0)==ord('q'):
    #     break


    face_inputs = (face_img.astype('float32') / 255).reshape((1, img_size, img_size, 3))
    pred_lmks = lmk_model.predict(face_inputs)[0].reshape((-1, 2))

    # compute landmark of original image
    new_lmks = ((pred_lmks - np.array([face_left, face_top])) / face_ratio).astype(np.int)
    ori_lmks = new_lmks + new_bb[0]

    # visualize
    cv2.rectangle(ori_img, pt1=tuple(ori_bb[0]), pt2=tuple(ori_bb[1]), color=(255, 255, 255), thickness=2)

    for i, l in enumerate(ori_lmks):
        cv2.putText(ori_img, str(i), tuple(l), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(ori_img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

    # debug. 원본이미지와 랜드마크
    # cv2.imshow('predict face part', ori_img)
    # print('press any key....(q is quit)')
    # if cv2.waitKey(0)==ord('q'):
    #     break

    # # wearing glasses
    glasses_center = np.mean([ori_lmks[0], ori_lmks[1]], axis=0)
    glasses_size = np.linalg.norm(ori_lmks[0] - ori_lmks[1]) * 2

    angle = -angle_between(ori_lmks[0], ori_lmks[1])
    M = cv2.getRotationMatrix2D((glasses.shape[1] / 2, glasses.shape[0] / 2), angle, 1)
    rotated_glasses = cv2.warpAffine(glasses, M, (glasses.shape[1], glasses.shape[0]))

    try:
        result_img = overlay_transparent(result_img, rotated_glasses, glasses_center[0], glasses_center[1],
                                         overlay_size=(
                                         int(glasses_size), int(glasses.shape[0] * glasses_size / glasses.shape[1])))
    except:
        print('failed overlay image')

    cv2.imshow('img', ori_img)
    cv2.imshow('result', result_img)
    filename, ext = os.path.splitext(f)
    # cv2.imwrite('result/%s_lmks%s' % (filename, ext), ori_img)
    # cv2.imwrite('result/%s_result%s' % (filename, ext), result_img)
    #
    k = cv2.waitKey(0)
    if k == ord('q'):
        break
    if k==ord('+'):
        scale += 0.1
        continue
    elif k==ord('-'):
        scale -= 0.1
        if scale <= 0:
            scale = 0.1
        continue
    fileindex+=1
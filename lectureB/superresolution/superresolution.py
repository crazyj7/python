'''
CSI의 확대 기능
'''

import os, cv2, glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import pyramid_reduce

plt.style.use('dark_background')

base_path='dataset'
img_base_path = os.path.join(base_path, "img_align_celeba")
target_img_path = os.path.join(base_path, "processed")

eval_list = np.loadtxt(os.path.join(base_path, 'list_eval_partition.csv'), dtype=str, delimiter=',', skiprows=1)

img_sample = cv2.imread(os.path.join(img_base_path, eval_list[0][0]))
h, w, _ = img_sample.shape

img = cv2.imread(img_path)
# 이미지 크기를 정사각형 크기로 잘라준다.
crop = img[int((h-w)/2):int(-(h-w)/2),:]




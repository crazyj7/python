'''
CAM 사진찍거나( q 입력) 이미지 파일에서
인물 추출하여 아웃포커싱하기.
추출만 10초 걸리는 문제...

'''


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize

import cv2
import time


# get_ipython().run_line_magic('matplotlib', 'inline')

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained models
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "../MODELS/mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# ## Configurations
# 
# We'll be using a models trained on the MS-COCO dataset. The configurations of this models are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.



class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# Create models object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The models classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below.
# The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# ## Run Object Detection


matplotlib.use('TkAgg')  ## 왜 시작할때 agg로 변경되었는지 모르겠음. 강제로 변경.
print('backend=', plt.get_backend())


bVideo = True
scaler = 1.0

if bVideo:
    cap = cv2.VideoCapture(0)
    print(cap.get(3), cap.get(4))
    # cap.set(3, 800)  # width
    # cap.set(4, 600)  # height
    if not cap.isOpened():
        print('video open failed.')
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('cap open failed.')
            break
        image = cv2.resize(frame, (int(frame.shape[1] * scaler), int(frame.shape[0] * scaler)))
        cv2.imshow('press q key', image)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
else:
    # Load a random image from the images folder
    # file_names = next(os.walk(IMAGE_DIR))[2]
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

    # filename = 'fashion.jpg'
    filename = 'girlgroup03.jpg'
    image = skimage.io.imread(os.path.join(IMAGE_DIR, filename))  # RGB color mode


# Run detection  ..  long time.....
print('detect object : start....')
t1 = time.time()
print('image shape=', image.shape)
results = model.detect([image], verbose=0)
t2 = time.time()
print('detect object : end....', t2-t1)
# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
# ## Extract Perrson from Mask
masks = r['masks'][:,:,r['class_ids']==1]
bHuman = False
print('masks shape', masks.shape)  # 480x640x n
print('classids', r['class_ids'])
print('scores', r['scores'])
print('human count=', np.sum(r['class_ids']==1))

if masks.shape[2]>0 :
    print('found person.. ', masks.shape[2])
    bHuman = True
    mask = np.sum(masks, axis=2).astype(np.bool)
    mask_3d = np.repeat(np.expand_dims(mask, axis=2), 3, axis=2).astype(np.uint8)
    plt.figure()
    plt.imshow(mask)
    plt.show()
    # print('mask=', mask.shape)
    # print('image=', image.shape)
    # cv2.imshow('mask', mask_3d)
else:
    print('not found')
# exit()

blursize = 25       # 블러링 중간.
# blursize = 35     # 블러링 강.

if bHuman:
    # ## Blurring
    blurred_img = cv2.GaussianBlur(image, (blursize,blursize), 0)       ## background bluring... (include person)
    mask_3d_blurred = (cv2.GaussianBlur(mask_3d*255, (101, 101), blursize, blursize)/255).astype(np.float32)

    person_mask = mask_3d_blurred * image.astype(np.float32)        ## person mask  with org img
    bg_mask = (1-mask_3d_blurred) * blurred_img.astype(np.float32)  ## background mask with blur img
    out = (person_mask+bg_mask).astype(np.uint8)

    # save output file.
    # plt.figure()
    # plt.title('org')
    # plt.imshow(image)
    # plt.savefig('output_org.png')
    # plt.close()
    cvimage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output_org.png', cvimage)

    # plt.figure()
    # plt.title('output')
    # plt.imshow(out)
    # plt.savefig('output_outfocus.png')
    # plt.close()
    cvout = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output_outfocus.png', cvout)

    # view mask and output
    plt.figure(figsize=(15,20))
    plt.subplot(2,2,1)
    plt.title('person mask')
    plt.imshow(person_mask.astype(np.uint8))

    plt.subplot(2,2,2)
    plt.title('bg mask')
    plt.imshow(bg_mask.astype(np.uint8))

    plt.subplot(2,2,3)
    plt.title('org')
    plt.imshow(image)

    plt.subplot(2,2,4)
    plt.title('output')
    plt.imshow(out)

    plt.show()






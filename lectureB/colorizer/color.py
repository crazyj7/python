
'''
흑백 사진을 컬러 사진으로 바꾸기
python color.py [mode] [file path]
mode = 1(high) or 2(low)
'''


import cv2 # opencv 3.4.2+ required
import os
import numpy as np
import matplotlib.pyplot as plt
import sys


if len(sys.argv)==3:
    mode = int(sys.argv[1])      # 1, 2
    image_path = sys.argv[2]    # image file path
elif len(sys.argv)==2:
    print('default mode=1')
    mode = 1
    image_path = sys.argv[1]
else:
    print('parameter : mode(1/2) imagepath')
    sys.exit()


proto = './models/colorization_deploy_v2.prototxt'

if mode==1:
    weights = './models/colorization_release_v2.caffemodel'
else:
    # colorization_release_v2_norebal.caffemodel is trained with a classification loss with no class re-balancing term.
    # The results are duller but "safer" colorizations
    weights = './models/colorization_release_v2_norebal.caffemodel'

# load cluster centers
pts_in_hull = np.load('./models/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)

# load model
net = cv2.dnn.readNetFromCaffe(proto, weights)
# net.getLayerNames()

# populate cluster centers as 1x1 convolution kernel
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]
# scale layer doesn't look work in OpenCV dnn module, we need to fill 2.606 to conv8_313_rh layer manually
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]


if True:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_input = img.copy()
    # convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_rgb = img.copy()

    # normalize input
    img_rgb = (img_rgb / 255.).astype(np.float32)

    # convert RGB to LAB
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    # only L channel to be used
    img_l = img_lab[:, :, 0]

    input_img = cv2.resize(img_l, (224, 224))
    input_img -= 50  # subtract 50 for mean-centering
else:
    img = cv2.imread(image_path)
    img_input = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plot images
# fig = plt.figure(figsize=(10, 5))
# fig.add_subplot(1, 2, 1)
# plt.imshow(img_rgb)
# fig.add_subplot(1, 2, 2)
plt.axis('off')
# plt.imshow(input_img, cmap='gray')
plt.imshow(img_rgb)
plt.show()

net.setInput(cv2.dnn.blobFromImage(input_img))
pred = net.forward()[0,:,:,:].transpose((1, 2, 0))

# resize to original image shape
pred_resize = cv2.resize(pred, (img.shape[1], img.shape[0]))

# concatenate with original image L
pred_lab = np.concatenate([img_l[:, :, np.newaxis], pred_resize], axis=2)

# convert LAB to RGB
pred_rgb = cv2.cvtColor(pred_lab, cv2.COLOR_Lab2RGB)
pred_rgb = np.clip(pred_rgb, 0, 1) * 255
pred_rgb = pred_rgb.astype(np.uint8)

# plot prediction result
fig = plt.figure(figsize=(20, 10))
fig.add_subplot(1, 2, 1).axis('off')
plt.imshow(img_l, cmap='gray')
fig.add_subplot(1, 2, 2).axis('off')
plt.imshow(pred_rgb)
plt.show()

# plt.savefig(output_filename)

# save result image file
filename, ext = os.path.splitext(image_path)
input_filename = '%s_input%s' % (filename, ext)
output_filename = '%s_output%s' % (filename, ext)

pred_rgb_output = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)

cv2.imwrite(input_filename, img_input)
cv2.imwrite(output_filename, np.concatenate([img, pred_rgb_output], axis=1))



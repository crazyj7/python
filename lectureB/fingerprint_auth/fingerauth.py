import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.models import Model, load_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa

import random


model = load_model('finger.h5')

x_real = np.load('dataset/x_real.npz')['data']
y_real = np.load('dataset/y_real.npy')
x_easy = np.load('dataset/x_easy.npz')['data']
y_easy = np.load('dataset/y_easy.npy')
x_medium = np.load('dataset/x_medium.npz')['data']
y_medium = np.load('dataset/y_medium.npy')
x_hard = np.load('dataset/x_hard.npz')['data']
y_hard = np.load('dataset/y_hard.npy')

print(x_real.shape, y_real.shape)

# plt.figure(figsize=(15, 10))
# plt.subplot(1, 4, 1)
# plt.title(y_real[0])
# plt.imshow(x_real[0].squeeze(), cmap='gray')
# plt.subplot(1, 4, 2)
# plt.title(y_easy[0])
# plt.imshow(x_easy[0].squeeze(), cmap='gray')
# plt.subplot(1, 4, 3)
# plt.title(y_medium[0])
# plt.imshow(x_medium[0].squeeze(), cmap='gray')
# plt.subplot(1, 4, 4)
# plt.title(y_hard[0])
# plt.imshow(x_hard[0].squeeze(), cmap='gray')

x_data = np.concatenate([x_easy, x_medium, x_hard], axis=0)
label_data = np.concatenate([y_easy, y_medium, y_hard], axis=0)

x_train, x_val, label_train, label_val = train_test_split(x_data, label_data, test_size=0.1)
print(x_data.shape, label_data.shape)
print(x_train.shape, label_train.shape)
print(x_val.shape, label_val.shape)

# new user fingerprint input
random_idx = random.randint(0, len(x_val))
random_img = x_val[random_idx]
random_label = label_val[random_idx]
seq = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0, 0.5)),
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-30, 30),
        order=[0, 1],
        cval=255
    )
], random_order=True)
random_img = seq.augment_image(random_img).reshape((1, 90, 90, 1)).astype(np.float32) / 255.

# matched image
match_key = random_label.astype(str)
match_key = ''.join(match_key).zfill(6)
print('match_key=', match_key)

label_real_dict = {}
for i, y in enumerate(y_real):
    key = y.astype(str)
    key = ''.join(key).zfill(6)
    label_real_dict[key] = i


rx = x_real[label_real_dict[match_key]].reshape((1, 90, 90, 1)).astype(np.float32) / 255.
ry = y_real[label_real_dict[match_key]]

pred_rx = model.predict([random_img, rx])

# unmatched image
unmatch_key, unmatch_idx = random.choice(list(label_real_dict.items()))

ux = x_real[unmatch_idx].reshape((1, 90, 90, 1)).astype(np.float32) / 255.
uy = y_real[unmatch_idx]

pred_ux = model.predict([random_img, ux])

plt.figure(figsize=(8, 4))
plt.subplot(1, 3, 1)
plt.title('Input: %s' %random_label)
plt.imshow(random_img.squeeze(), cmap='gray')
plt.subplot(1, 3, 2)
plt.title('O: %.02f, %s' % (pred_rx, ry))
plt.imshow(rx.squeeze(), cmap='gray')
plt.subplot(1, 3, 3)
plt.title('X: %.02f, %s' % (pred_ux, uy))
plt.imshow(ux.squeeze(), cmap='gray')

plt.show()


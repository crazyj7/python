
import cv2
import numpy as np


img1 = cv2.imread('fireball.png', cv2.IMREAD_UNCHANGED)
img2 = cv2.imread('fireball2.png', cv2.IMREAD_UNCHANGED)
img3 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
print(img1.shape)
print(img3.shape)
img4 = np.concatenate([img1, img3], axis=1)
cv2.imwrite('fireball_r.png', img4)
exit()




# concate

img1 = cv2.imread('plane.png', cv2.IMREAD_UNCHANGED)
img2 = cv2.imread('planeu.png', cv2.IMREAD_UNCHANGED)
img3 = cv2.imread('planed.png', cv2.IMREAD_UNCHANGED)

print(img1.shape)   # 55x89x4

imgcon = np.concatenate([img1, img2, img3], axis=1) # 55x267x4
print(imgcon.shape)

cv2.imwrite('plane_sp.png', imgcon)






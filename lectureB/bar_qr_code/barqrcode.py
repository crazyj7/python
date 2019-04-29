
'''

'''

import pyzbar.pyzbar as pyzbar
import cv2

img = cv2.imread('barcode1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
decoded = pyzbar.decode(img)
print(decoded)




img = cv2.imread('qr.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
decoded = pyzbar.decode(img)
print(decoded)






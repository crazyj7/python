'''
정면 얼굴만 감지함.

측면으로 각도가 틀어지거나, 일부가 안보이면 인식 실패.

'''

# import required libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ### Code - Haar Cascade Classifier
# XML training files for Haar cascade are stored in `opencv/data/haarcascades/` folder.
# First we need to load the required XML classifier. Then load our input image in grayscale mode.
# Many operations in OpenCV **are done in grayscale**.

# load cascade classifier training file for haarcascade
# haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_fullbody.xml')

# load test iamge
if False:
    test1 = cv2.imread('data/test1.jpg')
    # convert the test image to gray image as opencv face detector expects gray images
    gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);
    print('Faces found: ', len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(test1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(convertToRGB(test1))
    plt.show()


def detect_faces(f_cascade, colored_img, scaleFactor=1.1, minNeighbor=5):
    img_copy = np.copy(colored_img)
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    # let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbor);
    # go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_copy



if True:
    # debug
    test2 = cv2.imread('data/test5.jpg')
    faces_detected_img = detect_faces(haar_face_cascade, test2)
    plt.imshow(convertToRGB(faces_detected_img))
    plt.show()
    exit()


if False:
    # load another image
    test2 = cv2.imread('data/test3.jpg')
    faces_detected_img = detect_faces(haar_face_cascade, test2)
    plt.imshow(convertToRGB(faces_detected_img))
    plt.show()

    test2 = cv2.imread('data/test4.jpg')
    faces_detected_img = detect_faces(haar_face_cascade, test2)
    plt.imshow(convertToRGB(faces_detected_img))
    plt.show()

    # Well, we got two false positives. What went wrong there?
    # Remember, some faces may be closer to the camera and they would appear bigger than
    # those faces in the back. The scale factor compensates for this so can tweak that parameter.
    # For example, `scaleFactor=1.2` improved the results.
    test2 = cv2.imread('data/test4.jpg')
    faces_detected_img = detect_faces(haar_face_cascade, test2, scaleFactor=1.2)
    plt.imshow(convertToRGB(faces_detected_img))
    plt.show()


cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, img = cap.read()
    faces_detected_img = detect_faces(haar_face_cascade, img, 1.1, 5)
    # faces_detected_img = detect_faces(haar_face_cascade, img)
    cv2.imshow("face", faces_detected_img)

    if cv2.waitKey(1)==ord('q'):
        break

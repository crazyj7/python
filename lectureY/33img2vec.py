from PIL import Image
import numpy as np
import os, glob
from sklearn.model_selection import train_test_split

dirname = os.path.dirname(__file__)
os.chdir(dirname)

## test code. image to numpy array
with open('snow.jpg', 'rb') as file:
    img = Image.open(file)
    img = img.convert("RGB")
    img = img.resize((64,64))
    data = np.asarray(img)
    print(data)
    print(data.shape)


# 분류 대상
imagedir = ".\\images\\101"
category = ["chair", "camera", "butterfly"]
nb_classes = len(category)

image_w = 64
image_h = 64
pixels = image_h * image_w * 3

# load image data
x=[]
y=[]
for idx, cat in enumerate(category):
    # one-hot encoding. label
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    print(label)
    # image
    catpath = imagedir + '\\' + cat
    files = glob.glob(catpath+"/*.jpg")
    print(files)
    for i, f in enumerate(files):
        print(i, 'load ', f)
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        x.append(data)
        y.append(label)
x = np.array(x)
y = np.array(y)

print(x)
print(y)


## data split...
x_train, x_test, y_train, y_test = \
    train_test_split(x, y)
xy = (x_train, x_test, y_train, y_test)
np.save("./images/101.npy", xy, allow_pickle=True)
print("ok", len(y))


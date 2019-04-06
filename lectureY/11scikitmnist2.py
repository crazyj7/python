import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, svm, metrics


DATA_TRAIN_IMAGE = "train-images-idx3-ubyte"
DATA_TRAIN_LABEL = "train-labels-idx1-ubyte"
DATA_TEST_IMAGE = "t10k-images-idx3-ubyte"
DATA_TEST_LABEL = "t10k-labels-idx1-ubyte"

savepath = "./mnist"

test_label = np.load(savepath+"/"+DATA_TEST_LABEL+".npy")
test_image = np.load(savepath+"/"+DATA_TEST_IMAGE+".npy")

train_label = np.load(savepath+"/"+DATA_TRAIN_LABEL+".npy")
train_image = np.load(savepath+"/"+DATA_TRAIN_IMAGE+".npy")

print(test_label.shape, test_image.shape)
print(train_label.shape, train_image.shape)

# print(train_image[0])
#
# debug print text
if False:
    for idx in range(3):
        print('label=', train_label[idx])
        plain=train_image[idx].flatten()
        for i, v in enumerate(plain):
            print("{:3d}".format(v), end=" ")
            if i%28==27: print()

clf = svm.SVC()
# scaling ; 0~1
train_image = train_image/255
test_image = test_image/255

# long time... cut data.
train_image = train_image[0:2000]
train_label = train_label[0:2000]

# make input 2D
train_image2d = train_image.reshape(-1, 28*28)
test_image2d = test_image.reshape(-1, 28*28)

# train
clf.fit(train_image2d, train_label)

predict = clf.predict(test_image2d)

score = metrics.accuracy_score(test_label, predict)
print(score)



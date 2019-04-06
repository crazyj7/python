import urllib.request as req
import gzip, os, os.path
import struct
import numpy as np

DATA_TRAIN_IMAGE = "train-images-idx3-ubyte"
DATA_TRAIN_LABEL = "train-labels-idx1-ubyte"
DATA_TEST_IMAGE = "t10k-images-idx3-ubyte"
DATA_TEST_LABEL = "t10k-labels-idx1-ubyte"


savepath = "./mnist"
baseurl = "http://yann.lecun.com/exdb/mnist"
files = [
    DATA_TRAIN_IMAGE+".gz",
    DATA_TRAIN_LABEL+".gz",
    DATA_TEST_IMAGE+".gz",
    DATA_TEST_LABEL+".gz"
]

#download
if True:
    print('Phase1 : download mnist')
    if not os.path.exists(savepath): os.mkdir(savepath)
    for f in files:
        url = baseurl+"/"+f
        loc = savepath+"/"+f
        print('download:', url)
        if not os.path.exists(loc):
            req.urlretrieve(url, loc)
    # gunzip
    for f in files:
        gz_file = savepath+"/"+f
        raw_file = savepath+"/"+f.replace(".gz", "")
        print("gzip:", f)
        with gzip.open(gz_file, "rb") as fp:
            body = fp.read()
            with open(raw_file, "wb") as w:
                w.write(body)
    print("download & uncompress ok")


def change_data_labels(savepath, filename):
    file = savepath+"/"+filename
    # label file ; (4)magic (4)num (1)...
    labels=bytearray()
    with open(file, "rb") as fp:
        magic = fp.read(4)
        cnt = struct.unpack('>i', fp.read(4))
        # print('magic=', magic, 'cnt=', cnt[0])
        labels = fp.read(cnt[0])
        # print(labels)
        np_labels = np.frombuffer(labels, dtype=np.uint8)
        print('load train-label ok', cnt[0])
        np.save(file+".npy", np_labels)
        print('labels shape=', np_labels.shape)

def change_data_images(svaepath, filename):
    file = savepath+"/"+filename
    # image file ; (4) magic (4) count 28x28
    imglist=[]
    with open(file, "rb") as fp:
        magic = fp.read(4)
        cnt = struct.unpack('>i', fp.read(4))
        # print('magic=', magic, 'cnt=', cnt[0])
        cntrow = struct.unpack('>i', fp.read(4))
        print('row=', cntrow[0])
        cntcol = struct.unpack('>i', fp.read(4))
        print('col=', cntcol[0])
        for j in range(cnt[0]):
            imgdata = fp.read(28*28)
            np_imgdata = np.frombuffer(imgdata, dtype=np.uint8)
            # print(np_imgdata)
            np_imgdata = np_imgdata.reshape(28,28)
            imglist.append(np_imgdata)
        print('load train-image ok', cnt[0])
    imglist = np.asarray(imglist)
    np.save(file+".npy", imglist)
    print('imglist shape=', imglist.shape)

if True:
    print('Phase2: mnist to csv')

    file = savepath+"/train-labels-idx1-ubyte"
    # label file ; (4)magic (4)num (1)...
    labels=bytearray()
    change_data_labels(savepath, DATA_TEST_LABEL)
    change_data_images(savepath, DATA_TEST_IMAGE)
    change_data_labels(savepath, DATA_TRAIN_LABEL)
    change_data_images(savepath, DATA_TRAIN_IMAGE)






from PIL import Image
import numpy as np


# image to average hash
def average_hash(fname, size=16):
    img = Image.open(fname)
    img = img.convert('L')   # gray scale
    img = img.resize((size, size), Image.ANTIALIAS)
    pixel_data = img.getdata()
    pixels = np.array(pixel_data)

    pixels = pixels.reshape((size, size))
    # print(pixels)

    avg = pixels.mean()
    # print('mean=', avg)
    diff = 1 *(pixels > avg)        # 밝은 영역 마스킹 매트릭스
    # print('diff=', diff)
    return diff

def np2hash(ahash):
    bhash = []
    for nl in ahash.tolist():
        s1 = [str(i) for i in nl]
        s2 = "".join(s1)  # binary hash string.
        i = int(s2, 2)  # binary string to integer.
        bhash.append("%04x"%i)
    return "".join(bhash)

ahash1 = average_hash('night1.jpg')
print(ahash1)
print(np2hash(ahash1))
# 이미지가 조금 변경되어도 변경된 부분의 해시값도 조금 변경된다.
# 일반 해시랑은 다른 해시 모듈.


def getHamDist(file1, file2):
    ahash1 = average_hash(file1)
    ahash2 = average_hash(file2)
    hammingtundist = np.sum(ahash1.flatten()!=ahash2.flatten() )
    return hammingtundist

dist=getHamDist('night3.jpg', 'night2.jpg')
print( 'distance=', dist )

dist=getHamDist('night3.jpg', 'night4.jpg')
print( 'distance=', dist )




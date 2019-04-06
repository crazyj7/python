'''
이미지를 조작하여 여러가지 형태의 유사 이미지를 무한히 생성한다.
회전, 이동, 확대, 찌그러짐, 뒤집기 등...
'''
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

np.random.seed(3)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.7,
                                   zoom_range=[0.9, 2.2],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')
img = load_img('handwriting_shape/train/triangle/triangle001.png')
x = img_to_array(img)
x = x.reshape( (1,)+x.shape )
print(x)

i=0
for batch in train_datagen.flow(x, batch_size=1, save_to_dir='handwriting_shape/generate', save_prefix='gen_',
    save_format='png'):
    i+=1
    if i>30:
        break



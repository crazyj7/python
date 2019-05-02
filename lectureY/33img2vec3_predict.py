from keras.models import load_model
import numpy as np
from PIL import Image

target = 'test3.jpg'


## test code. image to numpy array
with open(target, 'rb') as file:
    img = Image.open(file)
    img = img.convert("RGB")
    img = img.resize((64,64))
    data = np.asarray(img)
    print(data)
    print(data.shape)

x = data
# normalize
x = x.astype("float") / 256
print(x.shape)

model = load_model('img2vec.models')
category = ["chair", "camera", "butterfly"]

x_test = np.array( [x] )
result = model.predict( x_test )
print(result)
idx = np.argmax(result[0])
print( category[idx] )

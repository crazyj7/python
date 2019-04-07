from PIL import Image
import numpy as np

with open('snow.jpg', 'rb') as file:
    img = Image.open(file)
    img = img.convert("RGB")
    img = img.resize((64,64))
    data = np.asarray(img)
    print(data)


from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import glob, os

# mdoelfile = 'models/20190501103342.h5'

filelist = glob.glob('models/*.h5')
modelfile = max(filelist, key=os.path.getctime)
print('modelfile=', modelfile)

x_val = np.load('dataset/x_val.npy').astype(np.float32)
y_val = np.load('dataset/y_val.npy').astype(np.float32)

model = load_model(modelfile)

y_pred = model.predict(x_val/255.)
y_pred_logical = (y_pred>0.5).astype(np.int)

print('test acc:%s'%accuracy_score(y_val, y_pred_logical))
cm = confusion_matrix(y_val, y_pred_logical)
sns.heatmap(cm, annot=True)
plt.show()


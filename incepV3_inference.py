from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import load_model
import numpy as np
import os
model = load_model('/home/peter/Downloads/keras_test/models/SGD-lr0.001_freeze200_dropout02/epoch73--test0.008776--val0.005489.hdf5')
direction = './data/train/NG_plastic/'
c= 0
l = os.listdir(direction)
for i in l:
    img_path = direction + i
    img = image.load_img(img_path,target_size=(300,300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    x= x /255
    y = model.predict(x)
    print(np.argmax(y))
    if np.argmax(y) != 0:
        c+=1
    #    print(y)
print(c,'/',len(l))
print(1-c/len(l))

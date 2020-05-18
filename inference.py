from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input

from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

model = load_model('Epoch96-loss0.06-val0.00.hdf5')
maping = ['plastic','wood','ok']
direction = './data/NG_plastic/'
c = 0
d =os.listdir(direction)
for i in d:
    img_path = direction + i
    img = image.load_img(img_path,target_size=(300,300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    x = x/255

    y = model.predict(x)
    res = (np.argmax(y))
    print(maping[res])
    if np.argmax(y) != 0:
        c+=1

    # img2 = cv2.imread(img_path)
    # cv2.imshow('',x)
    # cv2.waitKey(0)
print(c,len(d))
print(1-c/len(d))
    # print(y)

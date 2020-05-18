from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time
import os

cap = cv2.VideoCapture('2020-05-13-160116.webm')
model = load_model('Epoch96-loss0.06-val0.00.hdf5')
maping = ['NG_plastic','NG_wood','OK']

x1,x2 = 20,780
y1,y2 = 200,500
i = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret is True:
        i+=1

        crop4 = frame[y1:y2,x1:x2]
        crop4 = cv2.resize(crop4,(300,300))
        x = crop4[...,::-1].astype(np.float32)
        x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        x = x/250
        y = model.predict(x)
        res = (np.argmax(y))
        # print(y[0][res])

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX

    # Display the resulting frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.putText(frame,maping[res],(26,180), font, 1,(55,2,255),2,cv2.LINE_AA)
            cv2.imshow('frame',frame)
            cv2.waitKey(0)

        cv2.imshow('frame',frame)
    else:
        break   


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

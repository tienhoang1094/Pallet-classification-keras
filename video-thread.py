from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time
import os
from threading import Thread

cap = cv2.VideoCapture('/home/peter-linux/Desktop/AGF/Data-collector/video data/pallet-classes/8_May/2020-05-08-151720.webm')
model = load_model('/home/peter-linux/Desktop/AGF/test-modeltf2/pallet_model/Epoch86-loss0.00-val0.00.hdf5')
maping = ['NG_plastic','NG_wood','OK','Analyzing']

res = 3
def predict(frame):
    global res
    crop4 = frame[y1:y2,x1:x2]
    crop4 = cv2.resize(crop4,(224,224))
    x = crop4[...,::-1].astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = x/250
    y = model.predict(x)
    res = (np.argmax(y))

    
    
x1,x2 = 20,780
y1,y2 = 200,500
i = 0
time.sleep(2)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    i+=1
    time.sleep(0.02)

    if i % 3 == 0:
        x = Thread(target=predict, args=(frame,))
        x.start()

    cv2.putText(frame,maping[res],(26,180), cv2.FONT_HERSHEY_SIMPLEX , 1,(55,2,255),2,cv2.LINE_AA)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    # cv2.imshow('frame',frame)




# Display the resulting frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
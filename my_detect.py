from utils.datasets import *
from utils.utils import *
import cv2
import time
import os

cap = cv2.VideoCapture('test.mp4')
device = torch_utils.select_device('0')
torch.backends.cudnn.benchmark = True 

# Load model
weights = 'weights/best_yolo5s.pt'
model = torch.load(weights, map_location=device)['model']
model.to(device).eval()
names = model.names if hasattr(model, 'names') else model.modules.names
colors = [[0,255,0],[0,0,255],[0,0,255]]
print(names)
with torch.no_grad():
    while(True):

        ret, frame = cap.read()
        if ret:
            # Cropped processed frame
            process_frame = frame[200:500,20:780]
            cv2.rectangle(frame, (20,200), (780,500), [45, 255, 255], 1, cv2.LINE_AA)

            # Preprocess
            img = letterbox(process_frame, new_shape=640)[0]
            img = preprocess(img,device)

            # Predict
            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.6, 0.5,
                                   fast=True, classes=None, agnostic=False)
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], process_frame.shape).round()
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        xyxy = [int(xyxy[0]+20), int(xyxy[1]+200), int(xyxy[2]+20), int(xyxy[3]+200)]
                        plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            cv2.imshow('frame',frame)

        else:
            break
    cap.release()
    cv2.destroyAllWindows()

import sys
path = sys.path
# devel_path = '/home/mte/catkin_ws/devel/lib/python2.7/dist-packages'
# ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
# if devel_path in path:
#     path.remove(devel_path)
# if ros_path in path:
#     path.remove(ros_path)

import roslibpy

from utils.datasets import letterbox
from utils.utils import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

product_model = load_model('/home/peter-linux/Desktop/Pallet-classification-keras/Epoch109-loss0.001-val0.00003.h5')
product_mapping = ['CCL','Processed CCL','Nothing', 'Other']

check_pallet_req = False

def check_pallet_type_cb(msg):
    global check_pallet_req, check_cnt
    print('receive check request')
    check_pallet_req = True
    check_cnt = 0


def most_frequent(List):
    return max(set(List), key=List.count)


with torch.no_grad():
    cap = cv2.VideoCapture('test.mp4')
    x1, x2 = 20, 780  # Kich thuoc window
    y1, y2 = 200, 500
    cap.set(3, 800)  # Khung hinh (Khong thay doi)
    cap.set(4, 600)
    device = torch_utils.select_device('0')
    torch.backends.cudnn.benchmark = True 

    # Load model
    weights = 'weights/best_yolo5s.pt'
    print(weights)
    model = torch.load(weights, map_location=device)['model']
    model.to(device).eval()
    pallet_mapping = ['NG: 2 Side Pallet', 'NG: Wood Pallet', 'OK: 1 Side Pallet'] # model.names if hasattr(model, 'names') else model.modules.names
    colors = [[0,255,0],[0,0,255],[0,0,255]]

    check_cnt = 0
    pallet_result_list = []
    product_result_list = []
    pallet_final_result = -1
    product_final_result = -1
    check_continue_hotkey = False
    check_num = 1
    
    client = roslibpy.Ros(host='localhost', port=9090)
    # client.run()

    check_pallet_result_publisher = roslibpy.Topic(
        client, '/camera/pallet_type_result', 'std_msgs/String')
    check_pallet_type_subscriber = roslibpy.Topic(
        client, '/camera/check_pallet_type', 'std_msgs/String')
    check_pallet_type_subscriber.subscribe(check_pallet_type_cb)

    while(True):
        ret, frame = cap.read()
        if ret:
            # Hotkey
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                check_continue_hotkey = True
            if key != ord('r'):
                check_continue_hotkey = False

            # Cropped processed frame
            process_frame = frame[200:500,20:780]
            
            # Preprocess torch
            img = letterbox(process_frame, new_shape=640)[0]
            img = preprocess(img,device)

            # Preprocess tensoflow
            crop4 = cv2.resize(process_frame, (224, 224))
            x = crop4[..., ::-1].astype(np.float32)
            x = np.expand_dims(x, axis=0)
            x = x/250

            
            # Torch predict
            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.8, 0.5,
                                   fast=True, classes=None, agnostic=False)

            draw_border(frame,(20,200),(780,500),30,th=4)
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], process_frame.shape).round()
                    for *xyxy, conf, cls in det:
                        pallet_name = '%s %.2f' % (pallet_mapping[int(cls)], conf)  # Pallet type
                        pallet_final_result = int(cls)
                        xyxy = [int(xyxy[0]+20), int(xyxy[1]+200), int(xyxy[2]+20), int(xyxy[3]+200)]
                        distance = plot_one_box(xyxy, frame, label = pallet_name, color = colors[int(cls)], line_thickness=2) 

                        # Product
                        if xyxy[1]>280 and xyxy[3]<490:
                            if check_pallet_req or check_continue_hotkey:
                                # Tensorflow predict
                                y_product = product_model.predict(x)
                                res_product = (np.argmax(y_product)) # get max index
                                confident_product = (y_product[0][res_product]) # max value

                                font = cv2.FONT_HERSHEY_SIMPLEX

                                check_cnt += 1
                                product_result_list.append(res_product)
                                if check_cnt >= check_num:
                                    product_final_result = most_frequent(product_result_list)
                                    product_result_list = []
                                    print('product final result: ' + str(product_final_result))
                                    if client.is_connected:
                                        check_pallet_result_publisher.publish(
                                            roslibpy.Message({'data': str(pallet_final_result) + ',' + str(product_final_result)}))
                                        check_pallet_req = False
                                else:
                                    cv2.putText(frame, 'Analyzing...', (26, 180),
                                                font, 3, (0, 255, 255), 3, cv2.LINE_AA)

                            if check_cnt >= check_num:
                                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                check_cnt += 1
                                if pallet_final_result >= 0:
                                    pass
                                if product_final_result >= 0:
                                    cv2.putText(frame, product_mapping[res_product] + '-' + str(confident_product * 100)[:4] + '%', (26, 500), font, 2, (0, 255, 255), 3, cv2.LINE_AA)
                            if check_cnt >= 100:  # Continue display last result 5s
                                check_cnt = 0                            
            

            
            # Show frame
            cv2.imshow('frame',frame)
            # cv2.imshow('frame', cv2.resize(frame, (400, 300)))

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
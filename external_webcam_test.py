import cv2
import random
import  os
import numpy as np
from mobilenet_predictor import  Test_Graph
cam  = cv2.VideoCapture(0)



test = Test_Graph(model_file='./mobilenet_80_20_grayscale/mobilenet_new_dataset_gray_scale/intermediateintermediate_52500.pb')
print(cam.isOpened())
while(cam.isOpened()):

    ret, frame = cam.read()
    tf_frame =  cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    tf_frame  = test.read_tensor_from_image_image(image = tf_frame)
    # Our operations on the frame come here
    if ret:
        if frame is not None:

            # Display the resulting frame
            test.predict_currency(tf_frame)
            #print(currency[0],currency[1])
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1) & 0xff

            if(k == 27):
                break

        else:
            print('None image returned')
            break
    else:
        print('feame not returned')
        break

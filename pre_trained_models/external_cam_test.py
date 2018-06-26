import cv2
import random
import  os
import numpy as np
from mobilenet_predictor import Test_Graph
cam  = cv2.VideoCapture(2)

MAIN_INT = random.randint(0,1e6)
MAIN_INT = str(MAIN_INT) + '_'
dir = './prabhat_dataset/ten/'
#os.makedirs(dir)
i =0

print(cam.isOpened())


while(cam.isOpened()):

    ret, frame = cam.read()
    # Our operations on the frame come here
    if ret:
        if frame is not None:

            # Display the resulting frame
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1) & 0xff

            if (k == ord('k')):
                print(dir +MAIN_INT+ str(i) + '.png')
                cv2.imwrite(dir +MAIN_INT+ str(i) + '.png',frame)
                i = i + 1
            if(k == 27):
                break

        else:
            print('None image returned')
            break
    else:
        print('feame not returned')
        break





import cv2
import common
import random
import  os
import numpy as np
from mobilenet_predictor import Test_Graph
import tensorflow as tf

'''
cam  = cv2.VideoCapture(2)

test = Test_Graph(model_file='./model_graphs/mobilenet_80_20_grayscale_with_prahabt_images/mobilenet_new_dataset_gray_scale2/output_graph.pb')
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

'''


def read_tensor_from_image_image(image,
                                 input_height=224,
                                 input_width=224,
                                 input_mean=0,
                                 input_std=255):



    resized = cv2.resize(image, (input_height, input_width))
    float_caster = resized.astype(np.float32)
    resized = np.expand_dims(float_caster, 0)
    result = np.divide(np.subtract(resized, [input_mean]), [input_std])

    return result



if __name__=='__main__':

    cap  = common.VideoStream(2,mode='stream').start()
    # test = Test_Graph(model_file='./model_graphs/mobilenet_v2_140_new_dataset/output_graph.pb')
    test = Test_Graph(model_file='./model_graphs/mobilenet_80_20_grayscale_with_prahabt_images/mobilenet_new_dataset_gray_scale2/output_graph.pb')
    while not cap.stopped:
            frame = cap.read()
            if frame is not None:

                tf_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tf_frame = read_tensor_from_image_image(image=tf_frame)
                # frame = resizeImg(frame, (400, 400), keepAspect=True, padding=True)
                test.predict_currency(tf_frame)
                common.showImage(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
            else:
                print('No image returned')
                break



import tensorflow as tf
import numpy as np
import cv2
from tfRecord import Read_tf_record
import matplotlib.pyplot as plt


class Test_Graph(object):
    def __init__(self,model_file='./mobilenet_v2_140_new_dataset/output_graph.pb',label_file='./mobilenet_v2_140_new_dataset/output_labels.txt'):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with open(model_file, 'rb')    as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.labels = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            self.labels.append(l.rstrip())

        input_name = "Placeholder"
        output_name = 'final_result'
        self.input_operation = self.detection_graph.get_operation_by_name(input_name)
        self.output_operation = self.detection_graph.get_operation_by_name(output_name)

    def read_tensor_from_image_file(self,file_name,
                                    input_height=224,
                                    input_width=224,
                                    input_mean=0,
                                    input_std=255):
        input_name = "file_reader"
        file_reader = tf.read_file(file_name, input_name)
        if file_name.endswith(".png"):
            image_reader = tf.image.decode_png(
                file_reader, channels=3, name="png_reader")
        elif file_name.endswith(".gif"):
            image_reader = tf.squeeze(
                tf.image.decode_gif(file_reader, name="gif_reader"))
        elif file_name.endswith(".bmp"):
            image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
        else:
            image_reader = tf.image.decode_jpeg(
                file_reader, channels=3, name="jpeg_reader")
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)

        return result

    def read_tensor_from_image_image(self,image,
                                     input_height=224,
                                     input_width=224,
                                     input_mean=0,
                                     input_std=255):

        float_caster = tf.cast(image, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)
        sess.close()

        return result


    def predict_currency(self,image):
        t = image



        with tf.Session(graph=self.detection_graph) as sess:
            results = sess.run(self.output_operation.outputs[0], {
                self.input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        for i in top_k:
            print(self.labels[i], results[i])

    def predict_accuracy(self,path_to_tfRecord,):

        iterator = Read_tf_record().input_fn(path_to_tfRecord, batch_size=4,img_size=224)
        next_set = iterator.get_next()
        img_size = 224
        #image = tf.image.resize_images(x,[img_size,img_size])

        sess1 = tf.Session()
        sess2 = tf.Session(graph=self.detection_graph)

        sess1.run(iterator.initializer)
        no_of_examples = 0
        for record in tf.python_io.tf_record_iterator(path_to_tfRecord):
            no_of_examples += 1
        print('no of  examples', no_of_examples)
        total_sum = 0
        while (True):
            try:
                x_batch,y_batch = sess1.run(next_set)
                x_batch = self.read_tensor_from_image_image(x_batch)

                #print(x_batch.shape)

                #plt.show(plt.imshow(x_batch[0]))

                predicted = sess2.run(self.output_operation.outputs[0], {
                self.input_operation.outputs[0]:x_batch
            })
                #print(predicted,y_batch)
                sum = np.sum(np.argmax(predicted, 1) == np.argmax(y_batch, 1))

                total_sum = total_sum + sum
            except tf.errors.OutOfRangeError:
                break
        accuracy = np.float32(total_sum) / np.float32(no_of_examples)
        print('accuracy', 100*accuracy)

    def operation(self,image, per_cropped=0.2, hor_stretch=1.1, ver_stretch=1.1, rotation=20, brightness=0.1,
                  contrast=0.4):
        rows, cols = image.shape[0], image.shape[1]

        # rotation
        M1 = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
        rot_plus_20 = cv2.warpAffine(image, M1, (cols, rows))

        M2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), -1 * rotation, 1)
        rot_minus_20 = cv2.warpAffine(image, M2, (cols, rows))
        # Cropping part
        cropped = image[int(rows * per_cropped / 2):int(rows - rows * per_cropped / 2),
                  int(cols * per_cropped / 2):int(cols - cols * per_cropped / 2)]

        # Stretching part
        hor_img = cv2.resize(image, dsize=None, fx=hor_stretch, fy=1)
        ver_img = cv2.resize(image, dsize=None, fx=1, fy=ver_stretch)

        ## brightness part and contrast part

        bright1 = cv2.addWeighted(image, 1, image, 0, (brightness) * 255)
        bright2 = cv2.addWeighted(image, 1, image, 0, (brightness * 2) * 255)
        contrast1 = cv2.addWeighted(image, 1 + contrast, image, 0, 0)
        contrast2 = cv2.addWeighted(image, 1 + contrast, image, 0, -255 * contrast)
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        images = {
            "plus_20": rot_plus_20,
            "minus_20": rot_minus_20,
            "cropped": cropped,
            "hor_stretched": hor_img,
            "ver_stretched": ver_img,
            #"bright1": bright1,
            #"bright2": bright2,
            "contrast1": contrast1,
            "contrast2": contrast2,
            "gray_scale": grayscale
        }
        return images





if __name__=='__main__':
    test = Test_Graph(model_file='intermediateintermediate_34500.pb')


    #image = test.read_tensor_from_image_file(file_name='/home/pranav/Downloads/twenty2.jpeg')

    categories = ["plus_20", "minus_20", "cropped", "hor_stretched", "ver_stretched", "bright1", "bright2", "contrast1",
                  "contrast2", "gray_scale"]

    image  = cv2.imread('/home/pranav/Downloads/download (5).jpeg')
    # cv2.imshow('original',image)

    #image  = cv2.resize(image,(224,224))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)



    # plt.show(plt.imshow(image))
    #image = np.expand_dims(image,0)
    images = test.operation(image)




    #image = test.read_tensor_from_image_image(image=image)
    #image = test.read_tensor_from_image_image(image=dst1)


    for category in categories:
        #images[category] = cv2.resize(images[category],(224,224))
        #hsv_test = cv2.cvtColor(images[category],cv2.COLOR_RGB2HSV)
        #h = hsv_test[0,:,:]
        #print("h value",h)
        print("category",category)
        images[category] = cv2.cvtColor(images[category], cv2.COLOR_RGB2BGR)

        cv2.imshow(category,images[category])

        images[category] = cv2.cvtColor(images[category], cv2.COLOR_BGR2HSV)
        #image = test.read_tensor_from_image_image(images[category])

        h, _, _ = cv2.split(images[category])
        print("h value of image")
        cv2.imshow('h_'+category, h)
        cv2.waitKey(5000)
        #test.predict_currency(image=image)
        print("\n")


    

    plt.show(plt.imshow(image[0]))

    test.predict_currency(image=image)


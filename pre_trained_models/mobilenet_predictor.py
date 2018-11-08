import tensorflow as tf
import numpy as np
import cv2
from tfRecord import Read_tf_record,Write_tf_record
import random


class Test_Graph(object):
    def __init__(self, model_file = 'model_graphs/mobilenet_80_20_grayscale_with_prahabt_images/mobilenet_new_dataset_gray_scale2' ,
                 label_file='./model_graphs/output_labels.txt'):

        self.load_graph(model_file= model_file)
        self.labels = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            self.labels.append(l.rstrip())




    def load_graph(self, model_file):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with open(model_file, 'rb')    as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
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
        #print(result.max(),result.min())

        return result

    def read_tensor_from_image_image(self,image,
                                     input_height=224,
                                     input_width=224,
                                     input_mean=0,
                                     input_std=255):


        float_caster = tf.cast(image, tf.float32)
        #dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(float_caster, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)

        sess.close()
        '''

        resized = cv2.resize(image, (input_height, input_width))
        float_caster = resized.astype(np.float32)
        resized = np.expand_dims(float_caster, 0)
        result = np.divide(np.subtract(resized, [input_mean]), [input_std])
        '''




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
        print("\033[F"*6)
            #print("\r")


    def predict_accuracy(self,path_to_tfRecord = None,label_file = None, out_dir = None ,path_to_folder = None):

                    if path_to_folder is not None:
                        path_to_tfRecords =  Write_tf_record().write_tf_record(self, path_to_folder, label_file, out_dir)

                    elif(path_to_tfRecord is not None):
                        path_to_tfRecords = [path_to_tfRecord]

                    for path_to_tfRecord in path_to_tfRecords:
                        iterator = Read_tf_record().input_fn(path_to_tfRecord, batch_size=4, img_size=224)
                        next_set = iterator.get_next()

                        sess1 = tf.Session()
                        sess2 = tf.Session(graph=self.detection_graph)

                        sess1.run(iterator.initializer)
                        no_of_examples = 0
                        for record in tf.python_io.tf_record_iterator(path_to_tfRecord):
                            no_of_examples += 1
                            print('no of  examples', no_of_examples)
                            total_sum = 0
                            j = 0
                            while (True):
                                try:
                                    x_batch,y_batch = sess1.run(next_set)
                                    print(x_batch.shape)
                                    x_batch = self.read_tensor_from_image_image(x_batch)
                                    predicted = sess2.run(self.output_operation.outputs, {
                                    self.input_operation.outputs[0]:x_batch
                                })
                                    tmp = np.argmax(predicted[0], 1) == np.argmax(y_batch, 1)
                                    sum = np.sum(tmp)
                                    print(np.argmax(predicted[0], 1))



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



        images = {

            "plus_20": rot_plus_20,
            "minus_20": rot_minus_20,
            "cropped": cropped,
            "hor_stretched": hor_img,
            "ver_stretched": ver_img,
            "bright1": bright1,
            "bright2": bright2,
            "contrast1": contrast1,
            "contrast2": contrast2,
        }

        categories = ["plus_20", "minus_20", "cropped", "hor_stretched", "bright1", "bright2",
                      "ver_stretched", "contrast1", "contrast2"]

        random_index = random.sample(range(1, len(images)), 8)
        '''
        for index in random_index:
            images[categories[index]] = cv2.cvtColor(images[categories[index]], cv2.COLOR_RGB2GRAY)
        '''
        return images





if __name__=='__main__':

    graphs = ['/home/pranav/intermediateintermediate_3500.pb' , 'mobilenet_v2_140_new_dataset/output_graph.pb','mobilenet_80_20_grayscale/mobilenet_new_dataset_gray_scale/intermediateintermediate_52500.pb']
    test = Test_Graph(model_file='./model_graphs/mobilenet_80_20_grayscale_with_prahabt_images/mobilenet_new_dataset_gray_scale2/output_graph.pb')
    paths = ['currency_fifty_new.tfrecord','currency_five_hundred.tfrecord','currency_hundred.tfrecord'
             ,'currency_ten.tfrecord','currency_ten_new.tfrecord','currency_twenty.tfrecord','currency_two_hundred.tfrecord'
             ,'currency_two_thousand.tfrecord']
    paths2 = ['fifty.tfrecord','fifty new.tfrecord', 'five hundred.tfrecord', 'hundred.tfrecord'
        , 'ten.tfrecord', 'ten new.tfrecord', 'twenty.tfrecord',
             'two hundred.tfrecord','two thousand.tfrecord']

    paths3 = ['Fifty_Old','Fifty_New','Five_Hundred','Hundred','Ten_Old','Ten_new','Twenty','Two_Hundred','Two_Thousand']




    #test.predict_accuracy(path_to_tfRecord='./prabhat_dataset/' +'currency_five_hundred.tfrecord')


    for path in paths3:
        print("Path",path)
        test.predict_accuracy(path_to_tfRecord='/home/pranav/Downloads/NewDataset/NewDataset/'+path+'.tfrecord')

    '''
    #image = test.read_tensor_from_image_file(file_name='prabhat_dataset/hundred/2.png')
    #print(image.shape)

    image1 = cv2.cvtColor(image[0],cv2.COLOR_RGB2GRAY)
    image[:,:,:,0] = image1
    image[:,:,: ,1] = image1
    image[:,:,:, 2] = image1

    '''

    #categories = ["plus_20", "minus_20", "cropped", "hor_stretched", "bright1", "bright2","ver_stretched", "contrast1", "contrast2"]


    #image  = cv2.imread('/home/pranav/Downloads/download (5).jpeg')
    # cv2.imshow('original',image)

    image = test.read_tensor_from_image_file('/home/pranav/Downloads/ten1.jpeg')






    # plt.show(plt.imshow(image))
    #image = np.expand_dims(image,0)
    #print(image.shape)

    #images = test.operation(image[0])

    #image = test.read_tensor_from_image_image(image=image)
    #image = test.read_tensor_from_image_image(image=dst1)

    '''
    for category in categories:
        #images[category] = cv2.resize(images[category],(224,224))
        #hsv_test = cv2.cvtColor(images[category],cv2.COLOR_RGB2HSV)
        #h = hsv_test[0,:,:]
        #print("h value",h)
        print("category",category)
        #images[category] = cv2.cvtColor(images[category], cv2.COLOR_RGB2BGR)

        plt.show(plt.imshow(images[category]))



        image = test.read_tensor_from_image_image(images[category])

        test.predict_currency(image=image)
        print("\n")








    plt.show(plt.imshow(image[0]))

    test.predict_currency(image=image)

    '''

import tensorflow as tf
import numpy as np
from tflearn.data_utils import image_preloader
import  cv2
import sys
import os
import random
class Write_tf_record(object):
    def get_paths(self):
        MAIN_PATH = './dataset/'
        directories = ['ten', 'twenty', 'fifty', 'hundred', 'fivehundred', 'thousand']
        i = 0
        datas = []

        for directory in directories:
            path = os.path.join(MAIN_PATH, directory)
            files = os.listdir(path)
            for file in files:
                data = {
                    "path": os.path.join(path, file),
                    "label": i
                }
                datas.append(data)
            i = i + 1
        random.shuffle(datas)
        addrs = []
        labels = []
        for data in datas:
            addrs.append(data["path"])
            labels.append(data["label"])

        return [addrs,labels]

    def print_progress(self,count, total):
        # Percentage completion.
        pct_complete = float(count) / total

        # Status-message.
        # Note the \r which means the line should overwrite itself.
        msg = "\r- Progress: {0:.1%}".format(pct_complete)

        # Print it.
        sys.stdout.write(msg)
        sys.stdout.flush()
    def wrap_int64(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def wrap_bytes(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def convert(self,image_paths, labels, out_path):
        # Args:
        # image_paths   List of file-paths for the images.
        # labels        Class-labels for the images.
        # out_path      File-path for the TFRecords output file.

        print("Converting: " + out_path)

        # Number of images. Used when printing the progress.
        num_images = len(image_paths)
        # Open a TFRecordWriter for the output-file.
        with tf.python_io.TFRecordWriter(out_path) as writer:
            # Iterate over all the image-paths and class-labels.
            for i, (path, label) in enumerate(zip(image_paths, labels)):
                # Print the percentage-progress.
                self.print_progress(count=i, total=num_images - 1)

                # Load the image-file using matplotlib's imread function.
                img = cv2.imread(path)

                # Convert the image to raw bytes.
                img_bytes = img.tostring()

                # Create a dict with the data we want to save in the
                # TFRecords file. You can add more relevant data here.
                data = \
                    {
                        'image': self.wrap_bytes(img_bytes),
                        'label': self.wrap_int64(label)
                    }

                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)

                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)

                # Serialize the data.
                serialized = example.SerializeToString()

                # Write the serialized data to the TFRecords file.
                writer.write(serialized)


class Read_tf_record(object):
    def parse(self,serialized):
        # Define a dict with the data-names and types we expect to
        # find in the TFRecords file.
        # It is a bit awkward that this needs to be specified again,
        # because it could have been written in the header of the
        # TFRecords file instead.
        features = \
            {
                'image': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64)
            }

        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)

        # Get the image as raw bytes.
        image_raw = parsed_example['image']

        # Decode the raw bytes so it becomes a tensor with type.
        image = tf.decode_raw(image_raw, tf.uint8)

        # The type is now uint8 but we need it to be float.
        image = tf.cast(image, tf.float32)

        # Get the label associated with the image.
        label = parsed_example['label']

        # The image and label are now correct TensorFlow types.
        return image, label

    def input_fn(self,filenames, train, batch_size=32, buffer_size=2048):
            # Args:
            # filenames:   Filenames for the TFRecords files.
            # train:       Boolean whether training (True) or testing (False).
            # batch_size:  Return batches of this size.
            # buffer_size: Read buffers of this size. The random shuffling
            #              is done on the buffer, so it must be big enough.

            # Create a TensorFlow Dataset-object which has functionality
            # for reading and shuffling data from TFRecords files.
            dataset = tf.data.TFRecordDataset(filenames=filenames)

            # Parse the serialized data in the TFRecords files.
            # This returns TensorFlow tensors for the image and labels.
            dataset = dataset.map(self.parse)

            if train:
                # If training then read a buffer of the given size and
                # randomly shuffle it.
                dataset = dataset.shuffle(buffer_size=buffer_size)

                # Allow infinite reading of the data.
                num_repeat = None
            else:
                # If testing then don't shuffle the data.

                # Only go through the data once.
                num_repeat = 1

            # Repeat the dataset the given number of times.
            dataset = dataset.repeat(num_repeat)

            # Get a batch of data with the given size.
            dataset = dataset.batch(batch_size)

            # Create an iterator for the dataset and the above modifications.
            iterator = dataset.make_one_shot_iterator()

            # Get the next batch of images and labels.
            images_batch, labels_batch = iterator.get_next()

            # The input-function must return a dict wrapping the images.
            x = {'image': images_batch}
            y = labels_batch

            return x, y


class Train_Network(object):

    def __init__(self):
        self.img_size = 256
        self.num_classes = 6
        ####add sizes and numbers  to filters
        size_filter1 = 5
        size_filter2 = 5
        size_filter3 = 3
        size_filter4 = 3
        size_filter5 = 3

        no_filter1 = 64
        no_filter2 = 64
        no_filter3 = 32
        no_filter4 = 32
        no_filter5 = 16



        self.train_path ='./currency_train.tfrecords'
        self.test_path= './currency_test.tfrecords'

        self.filter1 = tf.get_variable("filter1",shape=[size_filter1 , size_filter1, 3 ,no_filter1],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        self.filter2 = tf.get_variable("filter2",shape=[size_filter2, size_filter2, no_filter1, no_filter2],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        self.filter3 = tf.get_variable("filter3", shape=[size_filter3, size_filter3, no_filter2, no_filter3], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        self.filter4 = tf.get_variable("filter4", shape=[size_filter4, size_filter4, no_filter3, no_filter4], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        self.filter5 = tf.get_variable("filter5", shape=[size_filter5, size_filter5, no_filter4, no_filter5], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())


        self.x = tf.placeholder(dtype=tf.float32,name='x',shape=[None,self.img_size,self.img_size,3])
        self.y = tf.placeholder(dtype=tf.float32,name='y',shape=[None,self.num_classes])


    def get_images(self,path):

       x, y = image_preloader(path, mode='file', image_shape=(self.img_size, self.img_size),
                                           categorical_labels=True, normalize=True)
       x = np.reshape(x, [len(x), self.img_size, self.img_size, 3])


       return [x,y]

    def create_network(self):



       ##############     CNN LAYERS #################
        ksize = [1,3,3,1]
        ## layer1 ##
        conv1 = tf.nn.conv2d(input=self.x,filter =self.filter1,strides= [1,1,1,1],
                             padding="SAME",name = "conv1")
        relu1 = tf.nn.relu(conv1,name="relu1")
        drop1 = tf.nn.dropout(relu1,keep_prob=1)
        pool1 = tf.nn.max_pool(drop1,ksize = ksize ,strides = [1,2,2,1],padding = 'VALID',name='pool1') ##add ksize


        ## layer2 ##
        conv2 = tf.nn.conv2d(input=pool1, filter=self.filter2, strides=[1, 1, 1, 1], padding="SAME",
                             name="conv2")
        relu2 = tf.nn.relu(conv2, name="relu2")

        drop2 = tf.nn.dropout(relu2, keep_prob=1)
        pool2 = tf.nn.max_pool(drop2, ksize=ksize, strides=[1, 2, 2, 1], padding='VALID', name='pool2')  ##add ksize

        ## layer3 ##
        conv3 = tf.nn.conv2d(input=pool2, filter=self.filter3, strides=[1, 1, 1, 1], padding="SAME",
                             name="conv3")
        relu3 = tf.nn.relu(conv3, name="relu3")

        drop3 = tf.nn.dropout(relu3, keep_prob=1)
        pool3 = tf.nn.max_pool(drop3, ksize=ksize, strides=[1, 2, 2, 1], padding='VALID', name='pool3')  ##add ksize

        ## layer4 ##
        conv4 = tf.nn.conv2d(input=pool3, filter=self.filter4, strides=[1, 1, 1, 1], padding="SAME",
                             name="conv4")
        relu4 = tf.nn.relu(conv4, name="relu4")

        drop4 = tf.nn.dropout(relu4, keep_prob=1)
        pool4 = tf.nn.max_pool(drop4, ksize=ksize, strides=[1, 2, 2, 1], padding='VALID', name='pool4')  ##add ksize

        ## layer5 ##
        conv5 = tf.nn.conv2d(input=pool4, filter=self.filter5, strides=[1, 1, 1, 1], padding="SAME",
                             name="conv5")
        relu5 = tf.nn.relu(conv5, name="relu5")

        drop5 = tf.nn.dropout(relu5, keep_prob=1)
        pool5 = tf.nn.max_pool(drop5, ksize=ksize, strides=[1, 2, 2, 1], padding='VALID', name='pool5')  ##add ksize



        ##############  fully connected layers   #################

        fc1 = tf.layers.flatten(pool5,name='fc1')

        self.W1 = tf.get_variable(dtype=tf.float32,name="W1",shape=[fc1.get_shape().as_list()[1] ,self.num_classes],initializer=tf.contrib.layers.xavier_initializer())  ##########  outputs ##########

        Z1 = tf.matmul(fc1,self.W1)

        self.y_predicted = tf.nn.softmax(Z1,name='y_predicted')



        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_predicted, 1)), dtype=tf.float32))

        self.sum = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_predicted, 1)), dtype=tf.float32),name="sum")

    def train_network(self):

        read_tfRecord = Read_tf_record()
        #x_train,y_train = self.get_images(self.train_path)
        #x_test,y_test = self.get_images(self.test_path)

        #########    hyperparameters    ########
        beta1 =0.9
        beta2 = 0.99
        epochs = 1000
        batch_size = 1

        save_path ='./training3/model.ckpt'
        log_path ='./training3/'

        learning_rate = 1e-5

        ########### end hyeperparameters #############
        self.cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,logits=self.y_predicted))
        loss_summary = tf.summary.scalar("loss", self.cost)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(self.cost)
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()


        with tf.Session() as sess:
            #saver = tf.train.import_meta_graph('./training2/model.ckpt-110')
            #saver.restore(sess, tf.train.latest_checkpoint('./training2/'))
            no_of_train_examples =1000
            no_of_test_examples = 10

            print("no of train examples")
            print(no_of_train_examples)
            no_of_train_batches = int(no_of_train_examples / batch_size)
            no_of_test_batches =int(no_of_test_examples / batch_size)
            writer = tf.summary.FileWriter(log_path, sess.graph)
            sess.run(init)
            for epoch in range (111,epochs+1):

                previous_train_batch = 0
                previous_test_batch = 0
                total_train_sum = 0
                total_test_sum = 0
                # Do our mini batches:
                for batch in range(no_of_train_batches):
                    print("epoch no: " + str(epoch))
                    x_train_batch,y_train_batch = read_tfRecord.input_fn(filenames=self.train_path,batch_size=batch_size,train=True)
                    print(sess.run(x_train_batch))
                    _, loss = sess.run([optimizer, self.cost],
                                       feed_dict={self.x: x_train_batch, self.y: y_train_batch})
                    sum = sess.run(self.sum, feed_dict={self.x: x_train_batch, self.y: y_train_batch})
                    total_train_sum = total_train_sum + sum
                    print("Loss: " + str(loss))
                if (epoch % 20 == 0):
                    saver.save(sess, save_path, global_step=epoch)
                    for batch in range(no_of_test_batches):
                        current_test_batch = previous_test_batch + batch_size

                        x_test_batch = x_test[previous_test_batch:current_test_batch]
                        y_test_batch = y_test[previous_test_batch:current_test_batch]
                        sum = sess.run(self.sum, feed_dict={self.x: x_test_batch, self.y: y_test_batch})
                        total_test_sum = total_test_sum + sum

                    self.train_accuracy = total_train_sum / no_of_train_examples
                    print("train_accuracy:" + str(self.train_accuracy))

                    self.test_accuracy = total_test_sum / no_of_test_examples
                    print("test_accuracy:" + str(self.test_accuracy))

                learning_rate = 1e-5 / (epoch * epoch + 1)
                










class Test_graph(object):


    def __init__(self):
        self.classes = ['ten', 'twenty', 'fifty', 'hundred', 'five hundred', 'thousand']
        path = './training2/currency_predictor-110.pb'
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with open(path, 'rb')    as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=detection_graph)
        self.y_predicted = detection_graph.get_tensor_by_name('y_predicted:0')
        self.x = detection_graph.get_tensor_by_name('x:0')
        self.y = detection_graph.get_tensor_by_name('y:0')
        self.batch_size =8


    def predict_currency(self,path):
        image = cv2.imread(path)
        image = cv2.resize(image,(256,256))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image,axis=0)
        prediction = self.sess.run(self.y_predicted,feed_dict={self.x:image})
        print(prediction)
        index =np.argmax(prediction,axis=1)

        print(self.classes[index[0]])

    

    def predict_accuracy(self):
        path = './test_data.txt'
        x_test, y_test = image_preloader(path, mode='file', image_shape=(256, 256),
                               categorical_labels=True, normalize=True)
        x_test= np.reshape(x_test, [len(x_test), 256, 256, 3])

        no_of_test_examples = x_test.shape[0]
        no_of_test_batches = int(no_of_test_examples / self.batch_size)
        sum = 0
        previous_batch = 0
        for batch in range(no_of_test_batches):
            current_batch = previous_batch + self.batch_size
            x_batch = x_test[previous_batch:current_batch]

            y_batch = y_test[previous_batch:current_batch]
            previous_batch = previous_batch + self.batch_size
            batch_predicted = self.sess.run(self.y_predicted, feed_dict={self.x: x_batch, self.y: y_batch})
            temp_sum = np.sum(np.argmax(batch_predicted,1) == np.argmax(y_batch,1))
            sum = sum + temp_sum
            print("sum",sum)


        accuracy = np.float32(sum / no_of_test_examples)
        print("accuracy during on the test set")
        print(accuracy * 100)


if __name__ == '__main__':

    network =Train_Network()
    network.create_network()
    network.train_network()


    
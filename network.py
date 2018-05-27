import tensorflow as tf
import numpy as np
from tflearn.data_utils import image_preloader
import  cv2




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
        no_filter3 = 64
        no_filter4 = 16
        no_filter5 = 16



        self.train_path ='./train_data.txt'
        self.test_path= './test_data.txt'

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
        drop1 = tf.nn.dropout(relu1,keep_prob=0.7)
        pool1 = tf.nn.max_pool(drop1,ksize = ksize ,strides = [1,2,2,1],padding = 'VALID',name='pool1') ##add ksize


        ## layer2 ##
        conv2 = tf.nn.conv2d(input=pool1, filter=self.filter2, strides=[1, 1, 1, 1], padding="SAME",
                             name="conv2")
        relu2 = tf.nn.relu(conv2, name="relu2")

        drop2 = tf.nn.dropout(relu2, keep_prob=0.7)
        pool2 = tf.nn.max_pool(drop2, ksize=ksize, strides=[1, 2, 2, 1], padding='VALID', name='pool2')  ##add ksize

        ## layer3 ##
        conv3 = tf.nn.conv2d(input=pool2, filter=self.filter3, strides=[1, 1, 1, 1], padding="SAME",
                             name="conv3")
        relu3 = tf.nn.relu(conv3, name="relu3")

        drop3 = tf.nn.dropout(relu3, keep_prob=0.7)
        pool3 = tf.nn.max_pool(drop3, ksize=ksize, strides=[1, 2, 2, 1], padding='VALID', name='pool3')  ##add ksize

        ## layer4 ##
        conv4 = tf.nn.conv2d(input=pool3, filter=self.filter4, strides=[1, 1, 1, 1], padding="SAME",
                             name="conv4")
        relu4 = tf.nn.relu(conv1, name="relu4")

        drop4 = tf.nn.dropout(relu4, keep_prob=0.7)
        pool4 = tf.nn.max_pool(drop4, ksize=ksize, strides=[1, 2, 2, 1], padding='VALID', name='pool4')  ##add ksize

        ## layer5 ##
        conv5 = tf.nn.conv2d(input=pool4, filter=self.filter5, strides=[1, 1, 1, 1], padding="SAME",
                             name="conv5")
        relu5 = tf.nn.relu(conv5, name="relu5")

        drop5 = tf.nn.dropout(relu5, keep_prob=0.7)
        pool5 = tf.nn.max_pool(drop5, ksize=ksize, strides=[1, 2, 2, 1], padding='VALID', name='pool5')  ##add ksize



        ##############  fully connected layers   #################

        fc1 = tf.layers.flatten(pool5,name='fc1')

        self.W1 = tf.get_variable(dtype=tf.float32,name="W1",shape=[fc1.get_shape().as_list()[1] ,self.num_classes],initializer=tf.contrib.layers.xavier_initializer())  ##########  outputs ##########

        Z1 = tf.matmul(fc1,self.W1)

        self.y_predicted = tf.nn.softmax(Z1,name='y_predicted')



        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_predicted, 1)), dtype=tf.float32))

        self.sum = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_predicted, 1)), dtype=tf.float32),name="sum")

    def train_network(self):

        x_train,y_train = self.get_images(self.train_path)
        x_test,y_test = self.get_images(self.test_path)

        #########    hyperparameters    ########
        beta1 =0.9
        beta2 = 0.99
        epochs = 200
        batch_size = 8

        save_path ='./training2/model.ckpt'
        log_path ='./training2/'

        learning_rate = 1e-5

        ########### end hyeperparameters #############
        self.cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,logits=self.y_predicted))
        loss_summary = tf.summary.scalar("loss", self.cost)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(self.cost)
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            no_of_train_examples = sess.run(tf.shape(x_train)[0])
            print("no of train examples")
            print(no_of_train_examples)
            no_of_batches = int(no_of_train_examples / batch_size)

            writer = tf.summary.FileWriter(log_path, sess.graph)
            sess.run(init)

            for epoch in range (epochs):
                
                previous_batch = 0
                # Do our mini batches:
                for batch in range(no_of_batches):
                    print("epoch no: " +str(epoch))
                    current_batch = previous_batch + batch_size
                    x_batch = x_train[previous_batch:current_batch]

                    y_batch = y_train[previous_batch:current_batch]
                    previous_batch = previous_batch + batch_size
                    print(x_batch.shape, current_batch - previous_batch)

                    _, loss = sess.run([optimizer, self.cost],
                                                  feed_dict={self.x: x_batch, self.y: y_batch})

                    print("Loss: " + str(loss))
                if(epoch % 10==0):
                        saver.save(sess, save_path, global_step=epoch)
                learning_rate = 1e-5/(epoch * epoch + 1)






class Test_graph(object):


    def __init__(self):
        self.classes = ['ten', 'twenty', 'fifty', 'hundred', 'five hundred', 'thousand']
        path = './training/currency_predictor.pb'
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


        accuracy = sum / no_of_test_examples
        print("accuracy during on the test set")
        print(accuracy * 100)


if __name__ == '__main__':
    network = Train_Network()
    network.create_network()
    network.train_network()

import  tensorflow as tf
import  cv2
import sys
import os
import random
class Write_tf_record(object):
    def get_paths(self,dir,label):
        MAIN_PATH = '/home/pranav/Downloads/NewDataset/NewDataset/'
        directories = dir

        datas = []
        path = os.path.join(MAIN_PATH, dir)

        files = os.listdir(path)
        for file in files:

                data = {
                    "path": os.path.join(path, file),
                    "label": label
                }
                print(data["path"],data["label"])
                datas.append(data)
            #i = i + 1
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
                try:
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

                    img_bytes = img.tostring()
                except:
                    print('exception')




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
    def write_tf_record(self,path_to_folder,label_file,out_dir):

        labels = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            labels.append(l.rstrip())

        label = 0
        tf_record_paths = []
        for dir in labels:
            print(dir)
            path = os.path.join(path_to_folder,dir)

            addrs, labels = Write_tf_record().get_paths(path, label)
            Write_tf_record().convert(image_paths=addrs, labels=labels,
                                      out_path=os.path.join(os.curdir(out_dir),dir) + '.tfrecord')
            tf_record_paths.append(os.path.join(os.curdir(out_dir),dir) + '.tfrecord')
            label = label + 1
        return tf_record_paths


class Read_tf_record(object):
    def parse(self,serialized):
        # Define a dict with the data-names and types we expect to
        # find in the TFRecords file.
        # It is a bit awkward that this needs to be specified again,
        # because it could have been written in the header of the
        # TFRecords file instead.
        features = \
            {
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            }

        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)

        # Get the image as raw bytes.
        image_raw = parsed_example['image']

        # Decode the raw bytes so it becomes a tensor with type.
        #image = tf.image.decode_image(image_raw, channels=3)

        image = tf.decode_raw(image_raw, tf.uint8)
        image = tf.reshape(image, [224, 224, 3])

        # The type is now uint8 but we need it to be float.
        #image = tf.cast(image, tf.float32)
        # Get the label associated with the image.
        label = parsed_example['label']
        label = tf.one_hot(label, depth=9)


        # The image and label are now correct TensorFlow types.
        return image, label

    def input_fn(self,filenames, batch_size,img_size, buffer_size=2048):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(Read_tf_record().parse)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        return(iterator)




if(__name__=='__main__'):
    dirs = ['Fifty_Old','Fifty_New','Five_Hundred','Hundred','Ten_Old','Ten_new','Twenty','Two_Hundred','Two_Thousand']
    #dirs = ['five_hundred']
    #dirs = os.listdir('/home/pranav/Downloads/NewDataset/NewDataset/')

    label = 0

    for dir in dirs:

        print(dir)

        addrs,labels = Write_tf_record().get_paths(dir,label)
        # Write_tf_record().convert(image_paths=addrs,labels=labels,out_path='./new_currency_tfrecords/currency_'+dir+'.tfrecord')
        Write_tf_record().convert(image_paths=addrs, labels=labels,out_path='/home/pranav/Downloads/NewDataset/NewDataset/'+dir+'.tfrecord')
        label = label + 1





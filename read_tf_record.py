import tensorflow as tf




train_path = '/home/pranav/PycharmProjects/Note_classifier/currency_train.tfrecords'
x=y=None
for i in range(200):
    x_prev=x
    y_prev =y
    x,y = input_fn(train_path,True)
    print(x == x_prev)
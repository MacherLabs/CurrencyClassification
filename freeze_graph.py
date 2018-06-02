import tensorflow as tf



saver = tf.train.import_meta_graph('/home/pranav/training2/model.ckpt-980.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "/home/pranav/training2/model.ckpt-980")


output_node_names="y_predicted,x,y"
output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session
            input_graph_def, # input_graph_def is useful for retrieving the nodes 
            output_node_names.split(",")  
)

output_graph="./currency_predictor-980.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
 
sess.close()


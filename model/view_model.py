import tensorflow as tf  
import os  


log_dir = 'inception_log'  
if not os.path.exists(log_dir):  
    os.makedirs(log_dir)  
    

inception_graph_def_file = 'inception_v3_2016_08_28_frozen.pb'
with tf.Session() as sess:  
    with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read())  
        tf.import_graph_def(graph_def, name='')  
    writer = tf.train.SummaryWriter(log_dir, sess.graph)  
    writer.close()  

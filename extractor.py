from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import csv
import pickle
from utils import ImageLoader


# training data dir
data_dir = '../image_captioning/train/images'


class ImageFeatureExtractor(object):

    def __init__(self, model_path):
        """Load TensorFlow CNN model."""
        assert os.path.exists(model_path), 'File does not exist %s' % model_path
        self.model_path = model_path
        # load graph
        with tf.gfile.FastGFile(os.path.join(model_path), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        # create a session for feature extraction
        self.session = tf.Session()
        self.writer = None

    def extract_features(self, image, tensor_name='InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0'):
        """Extract image feature from image (numpy array) or from jpeg file."""
        sess = self.session
        feat_tensor = sess.graph.get_tensor_by_name(tensor_name)
        # image is a path to an jpeg file
        assert os.path.exists(image), 'File does not exist %s' % image
        image_loader = ImageLoader()
        features = sess.run(feat_tensor, {'input:0': image_loader.load_imgs([image])})
        return list(np.squeeze(features))


if __name__ == '__main__':
    ife = ImageFeatureExtractor('model/inception_v3_2016_08_28_frozen.pb')
    anns = csv.reader(open("train/anns.csv"))
    count = 0
    img_2048_dict = {}
    for row in anns:
        try:
            if row[1] == 'caption':
                continue
            image_path = row[2]
            img_2048_dict[int(row[3])] = ife.extract_features(image_path)
            count += 1
            if count % 100 == 0:
                print('%d finished' % count)
            if count % 10000 == 0:
                with open('./train/train_img2048_%d.pkl' % (count / 10000),
                          'wb') as f:
                    pickle.dump(img_2048_dict, f)
                img_2048_dict = {}
        except:
            continue
    with open('./train/train_img2048_%d.pkl' % (count / 10000 + 1),
              'wb') as f:
        pickle.dump(img_2048_dict, f)
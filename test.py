from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pickle
import numpy as np
from extractor import ImageFeatureExtractor
from model import image_caption_model


if __name__ == '__main__':
    max_sent_len = 28
    model_path = './weights/v1.0.0_11_0_1494239663.5093253_602.h5'
    image_path = sys.argv[1]
    ife = ImageFeatureExtractor('model/inception_v3_2016_08_28_frozen.pb')

    with open('./train/word2idx.pkl', 'rb') as f:
        word2idx = pickle.load(f)
    with open('./train/idx2word.pkl', 'rb') as f:
        idx2word = pickle.load(f)
    vocab_size = len(word2idx) + 1
    model = image_caption_model(vocab_size=vocab_size)
    model.load_weights(model_path)
    start_sign = word2idx['+']

    img = np.array([ife.extract_features(image_path)])

    cur, vhist, answer = np.array([[start_sign]]), np.array([[0] * vocab_size]), []
    vhist = np.array(vhist)
    for idx in range(0, max_sent_len):
        seq = np.array([[1 if i == idx else 0 for i in range(0, max_sent_len)]])
        out = model.predict([img, cur, seq, vhist])[0]
        nxt = int(np.argmax(out))
        ans = idx2word.get(nxt, '<?>')
        print(ans, 'score:', out[nxt])
        cur = np.array([[nxt]])
        tmp_vhist = np.array([[0] * vocab_size])
        tmp_vhist[0, nxt] = 1
        vhist = np.array([np.logical_or(vhist[0, :], tmp_vhist[0, :])])
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pickle

import csv
import numpy as np

from model import image_caption_model
from joblib import Parallel, delayed
import time
from keras.utils.layer_utils import print_summary


def gene_word_caption(captions):
    word2idx = dict()
    idx2word = dict()
    word_count = 0

    ret = []
    for caption in captions:
        caption = caption.replace(',', ' ,')
        caption = caption.replace(';', ' ;')
        caption = caption.replace(':', ' :')
        caption = caption.replace('.', ' .')
        tokens = caption.split(' ')
        res = []
        for word in tokens:
            if word == '':
                continue
            if word not in word2idx:
                word2idx[word] = word_count
                idx2word[word_count] = word
                word_count += 1
            res.append(word2idx[word])
        ret.append(res)

    # add start sign
    word2idx['+'] = word_count
    idx2word[word_count] = '+'
    word_count += 1

    return ret, word2idx, idx2word, word_count


def gen_batch_in_thread(img_map, df_cap, vocab_size, n_jobs=4,
                        size_per_thread=32):
    imgs, curs, nxts, seqs, vhists = [], [], [], [], []
    returns = Parallel(n_jobs=4, backend='threading')(
                            delayed(generate_batch)
                            (img_train, df_cap, vocab_size, size=size_per_thread)
                            for i in range(0, n_jobs))

    for triple in returns:
        imgs.extend(triple[0])
        curs.extend(triple[1])
        nxts.extend(triple[2])
        seqs.extend(triple[3])
        vhists.extend(triple[4])

    return np.array(imgs), np.array(curs).reshape((-1, 1)), np.array(nxts), \
        np.array(seqs), np.array(vhists)


def generate_batch(img_map, df_cap, vocab_size, size=32, max_caplen=28):
    imgs, curs, nxts, seqs, vhists = [], [], [], [], []

    for idx in np.random.randint(len(df_cap), size=size):
        row = df_cap[idx]
        cap = row['caption']

        img = img_map[row['img_id']]

        vhist = np.zeros((len(cap)-1, vocab_size))

        for i in range(1, len(cap)):
            seq = np.zeros((max_caplen))
            nxt = np.zeros((vocab_size))
            nxt[cap[i]] = 1
            curs.append(cap[i-1])
            seq[i-1] = 1

            if i < len(cap)-1:
                vhist[i, :] = np.logical_or(vhist[i, :], vhist[i-1, :])
                vhist[i, cap[i-1]] = 1

            nxts.append(nxt)
            imgs.append(img)
            seqs.append(seq)

        vhists.extend(vhist)

    return imgs, curs, nxts, seqs, vhists


if __name__ == '__main__':
    # initialization
    n_jobs = 64
    size_per_thread = 64
    mdl_path = 'weights/'

    pkl_count = 0
    img_train = {}
    while True:
        pkl_count += 1
        try:
            with open('train/train_img2048_%d.pkl' % pkl_count, 'rb') as f:
                img_train.update(pickle.load(f))
        except FileNotFoundError:
            break
    print('got %d image context vectors' % len(img_train))

    anns = csv.reader(open("train/anns.csv"))
    anns = [row for row in anns if row[1] != 'caption']
    print('got %d training captions' % len(anns))

    captions = ['+ ' + row[1] for row in anns]
    captions, word2idx, idx2word, vocab_size = gene_word_caption(captions)
    print('got %d words, saving word translate dict...' % vocab_size)
    with open('./train/word2idx.pkl', 'wb') as f:
        pickle.dump(word2idx, f)
    with open('./train/idx2word.pkl', 'wb') as f:
        pickle.dump(idx2word, f)

    img_ids = [int(row[3]) for row in anns]

    df_cap = []
    for i in range(0, len(img_ids)):
        df_cap.append(
            dict(img_id=img_ids[i], caption=captions[i])
        )

    model = image_caption_model(vocab_size=vocab_size)

    if len(sys.argv) >= 2:
        print('load weights from : {}'.format(sys.argv[1]))
        model.load_weights(sys.argv[1])

    # insert ur version name here
    version = 'v1.0.0'
    batch_num = 70
    hist_loss = []

    for i in range(0, 100):
        for j in range(0, batch_num):
            s = time.time()
            img1, cur1, nxt1, seq1, vhists1 = gen_batch_in_thread(
                img_train, df_cap, vocab_size, n_jobs=n_jobs,
                size_per_thread=size_per_thread)
            hist = model.fit([img1, cur1, seq1, vhists1], nxt1,
                             batch_size=n_jobs * size_per_thread,
                             nb_epoch=1, verbose=0, shuffle=True)

            print("epoch {0}, batch {1} - training loss : {2}".format(
                i, j, hist.history['loss'][-1]))
            # record the training history
            hist_loss.extend(hist.history['loss'])

            if j % int(batch_num / 2) == 0:
                print('check point arrived, saving...')
                m_name = "{0}{1}_{2}_{3}_{4}_{5}.h5".format(
                    mdl_path, version, i, j, time.time(), str(vocab_size))
                model.save_weights(m_name)

# Image Caption

This is a simple implementation of `Image Caption` trained on MS COCO dataset.

The project is based on these repos:

[LemonATsu/Keras-Image-Caption](https://github.com/LemonATsu/Keras-Image-Caption)

[DeepRNN/image_captioning](https://github.com/DeepRNN/image_captioning)

## environments

system: win10 x64

cuda version: 8.0

cudnn version: 5.1

You should also have a GPU card with 4GB or larger graph memory. Nvidia GTX 1060+ is recommended.

## requirements

```
joblib==0.11
numpy==1.12.1
tensorflow-gpu==0.12.0
keras==1.2.2
```

## How to use

### prepare inception_v3 model

Download [inception_v3_2016_08_28_frozen.pb](https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz) and unpack it to `model/inception_v3_2016_08_28_frozen.pb`

### prepare COCO training data

Download [COCO 2014 Training images [80K/13GB]](http://msvocds.blob.core.windows.net/coco2014/train2014.zip) dataset and unpack all training jpg files to `train/images/`

### prepare anns.csv

The `anns.csv` is a table contains training images' path and their captions. When training, **ONLY** captions in `anns.csv` will be used.

We provide a default `anns.csv` contains about 56K captionss. You can generate this file on your own.

### extract image features

Run
```
python extractor.py
```
to generate image features.

**Warning:** `pickle.dump` method in python will cost a large amount of memory.

### fix a keras bug

When using tensorflow as keras backend, Maybe You should modify `keras/optimizers.py` like [this](https://github.com/fchollet/keras/pull/4915/files).

### train

Run
```
python train.py
```
to train the models. Checkpoint file will be save to `weights/`.

### test

Modify `model_path` to checkpoint file you have got and run
```
python test.py path/to/test/image.jpg
```
to get the result.

## TODO

+ add val data

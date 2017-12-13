# Faster-RCNN_TF

Faster-RCNN_TF on synthetic text data

Fork of this [repo](https://github.com/smallcorgi/Faster-RCNN_TF).

This is an experimental Tensorflow implementation of Faster RCNN - a convnet for object detection with a region proposal network.
For details about R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

### Requirements: software

1. Requirements for Tensorflow (see: [Tensorflow](https://www.tensorflow.org/))

2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository

2. Build the Cython modules
    ```Shell
    cd Faster-RCNN_TF/lib/
    make
    ```




### References
[Faster R-CNN caffe version](https://github.com/rbgirshick/py-faster-rcnn)

[A tensorflow implementation of SubCNN (working progress)](https://github.com/yuxng/SubCNN_TF)
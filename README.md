# Faster-RCNN_TF

Faster-RCNN_TF on synthetic text data

Fork of this [repo](https://github.com/smallcorgi/Faster-RCNN_TF).

This is an experimental Tensorflow implementation of Faster RCNN - a convnet for object detection with a region proposal network.
For details about R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

### Generate data

You can either generate data from the command line or in a python script using the functions in `generate_data/` folder.

#### Using the command line

The following will generate one image and its corresponding xml in the `generate_data/` folder

```
cd generate_data/
python main.py
```

Also check out  

```
python main.py --help
```

To see available options for tuning:

- Number of images to generate
- Number of symbols per images
- Path where to put the .xml
- Path where to put the .jpeg
- Maximum overlap btw images

For instance the following command would generate 10 images with at most 5 symbols per image and put both at the project root

```
python main.py --n_images 10 --n_samples_per_image 5 --filepath '../' 
```

### References
[Faster R-CNN caffe version](https://github.com/rbgirshick/py-faster-rcnn)

[A tensorflow implementation of SubCNN (working progress)](https://github.com/yuxng/SubCNN_TF)
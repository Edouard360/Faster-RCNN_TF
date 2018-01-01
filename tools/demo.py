import tensorflow as tf
from config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
from config import TEXT_CLASSES
import os, sys, cv2
import argparse
from customNet import CustomNet
from vis_detections import vis_detections



CLASSES = TEXT_CLASSES

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = 'data/demo/'+image_name
    im = cv2.imread(im_file,0)[:,:,np.newaxis]

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, rpn_cls_score = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    #im = im[:, :, (2, 1, 0)] # for 3 channels only
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im[:,:,0], aspect='equal',cmap='gray')

    CONF_THRESH = 0.9
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = CustomNet( state='TEST')
    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, 'models/weights/iter_12000.ckpt')
   
    #sess.run(tf.initialize_all_variables())



    im_names = ['050.jpg','051.jpg','052.jpg']#,'001.jpg','002.jpg','010.jpg'] # Let's see if we can overfit


    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(sess, net, im_name)

    plt.show()


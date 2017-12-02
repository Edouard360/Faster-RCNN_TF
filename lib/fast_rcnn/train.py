# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
#import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
#from gt_data_layer.layer import GtDataLayer
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time
import matplotlib.pyplot as plt

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, saver, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb) #TODO
        print 'done'

        # For checkpoint
        self.saver = saver

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG: #and net.layers.has_key('bbox_pred'):
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG: #and net.layers.has_key('bbox_pred'):
            print "yes"
            with tf.variable_scope('bbox_pred', reuse=True):
                # restore net to original state
                sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
                sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})

    def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul


    def train_model(self, sess, max_iters):
        """Network training loop."""
        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

        # RPN
        # classification loss
        with tf.name_scope('rpn'):
            with tf.name_scope('rpn_cross_entropy'):
                # self.net.get_output('rpn_cls_score_reshape')
                rpn_cls_score = tf.reshape(self.net.rpn_cls_score,[-1,2])
                rpn_label = self.net.rpn_labels
                # tf.logical_and(tf.not_equal(rpn_label,-1),tf.not_equal(rpn_label,0))
                rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2])
                rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1])
                tf.summary.histogram('rpn_cls_score',rpn_cls_score)
                rpn_fg_acc=tf.reduce_mean(tf.cast(tf.equal(tf.greater_equal(rpn_cls_score[:,0],0.5), tf.equal(rpn_label,0)),tf.float32))
                tf.summary.scalar('rpn_fg_acc', rpn_fg_acc)
                #class_weights = tf.Variable(trainable=False,initial_value=[[1,1]],dtype=tf.int32)
                #rpn_label_weighted = tf.reduce_sum(tf.multiply(rpn_label, class_weights),1)
                #sparse_
                rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
                tf.summary.scalar('cross_entropy', rpn_cross_entropy)
            with tf.name_scope('rpn_cls_score'):
                rpn_smooth_l1 = self._modified_smooth_l1(3.0, self.net.rpn_bbox_pred, self.net.rpn_bbox_targets, self.net.rpn_bbox_inside_weights, self.net.rpn_bbox_outside_weights)
                rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))
            rpn_loss = rpn_cross_entropy + rpn_loss_box
            tf.summary.scalar('rpn_loss',rpn_loss)


        # R-CNN
        with tf.name_scope('rcnn'):
            with tf.name_scope('cls'):
                labels = tf.reshape(self.net.labels,[-1])
                cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.net.logits, labels=labels))

                bg_acc = tf.reduce_mean(
                    tf.cast(tf.equal(tf.greater_equal(self.net.cls_prob[:, 0], 0.5), tf.equal(labels, 0)), tf.float32))
                cl1_acc = tf.reduce_mean(
                    tf.cast(tf.equal(tf.greater_equal(self.net.cls_prob[:,1], 0.5), tf.equal(labels, 1)), tf.float32))
                tf.summary.scalar('bg_acc',bg_acc)
                tf.summary.scalar('cl1_acc', cl1_acc)
            with tf.name_scope('bbox'):
                smooth_l1 = self._modified_smooth_l1(1.0, self.net.bbox_pred, self.net.bbox_targets, self.net.bbox_inside_weights, self.net.bbox_outside_weights)
                loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]))
                tf.summary.scalar('loss_box', loss_box)
            rcnn_loss = cross_entropy + loss_box
            tf.summary.scalar('rcnn_loss', rcnn_loss)


        with tf.name_scope('total_loss'):
            loss = rpn_loss+rcnn_loss
            tf.summary.scalar('loss', loss)

        train_op = tf.train.AdamOptimizer(5e-4).minimize(loss)

        merge = tf.summary.merge_all()
        name='example_name_performance' # should include hyperparameters
        train_writer = tf.summary.FileWriter('../tmp/'+name,sess.graph)


        # iintialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, self.saver, True)

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            blobs = data_layer.forward()

            feed_dict={self.net.data: blobs['data'], self.net.im_info: blobs['im_info'], self.net.keep_prob: 0.5, \
                           self.net.gt_boxes: blobs['gt_boxes']}


            timer.tic()

            # loss_cls_value, loss_box_value --- cross_entropy, loss_box,
            debug_info, summary, rpn_fg_acc_value, rpn_loss_cls_value, rpn_loss_box_value, _ = sess.run([self.net.debug_info,merge,rpn_fg_acc,rpn_cross_entropy, rpn_loss_box, train_op], #cross_entropy, loss_box,
                                                                                                feed_dict=feed_dict)

            train_writer.add_summary(summary, iter) #TODO : uncomment when clean

            #print("Verify shape",rpn_cls_score_value.shape)

            # Classification repartition

            # if prev is not None and prev.shape==rpn_cls_score_value.shape:
            #     print (rpn_cls_score_value==prev).mean()
            #prev = rpn_cls_score_value

            # cls_score = [(rpn_cls_score_value[0, :, :, 2 * k] >= rpn_cls_score_value[0, :, :, 2 * k + 1]) for k in range(9)]
            # cls_or_not_score = [(rpn_cls_score_value[0, :, :, 2 * k] >= rpn_cls_score_value[0, :, :, 2 * k + 1]).mean() for k in range(9)]
            # ax.clear()
            # ax.imshow(cls_score[5] + 0, aspect='equal')

            timer.toc()
            #print 'Debug info:\ntotal_anchors %.0f\ntotal_rpn %.0f\nbg %.0f\nfg %.0f'%tuple(debug_info)
            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                # print("Mean classification :",
                #       (rpn_cls_score_value[:, 0] <= rpn_cls_score_value[:, 1]).mean())  # Proportion of the foreground
                print "RPN value classif.",rpn_fg_acc_value
                print 'iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f'%\
                        (iter+1, max_iters, rpn_loss_cls_value+rpn_loss_box_value,rpn_loss_cls_value,rpn_loss_box_value)
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

                # , loss_cls: %.4f, loss_box: %.4f, lr: %f'%\
                # + rpn_loss_box_value + loss_cls_value + loss_box_value ,rpn_loss_cls_value, ,loss_cls_value, loss_box_value, lr.eval()

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    # if cfg.TRAIN.USE_FLIPPED:
    #     print 'Appending horizontally-flipped training examples...'
    #     imdb.append_flipped_images()
    #     print 'done'

    print 'Preparing training data...'
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            pass#gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            pass#layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    roidb = filter_roidb(roidb)
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'

import tensorflow as tf
from rpn_msr.anchor_target_layer_tf import anchor_target_layer
from rpn_msr.proposal_layer_tf import proposal_layer
from rpn_msr.proposal_target_layer_tf import proposal_target_layer
from networks.network import Network
import roi_pooling_layer.roi_pooling_op as roi_pool_op
from utils.smooth_l1 import smooth_l1
from utils.adapt_rois import adapt_rois
import tensorflow.contrib.layers as layers
from roi_pooling_layer_2.roi_pooling_layer import roi_pooling_op_2
# layers.conv2d #
#define

n_classes =  3
_feat_stride = [8,] # _feat_stride  generer trop d'anchor est ultra long
anchor_scales = [10]#[8, 16, 32] -> car 7 * 16 = 112 ~ 100 => en plus en divisant par 16 ca tombera rond

class CustomNet(Network):
    def __init__(self, trainable=True,state='TRAIN'):
        self.state=state
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes})
        self.trainable = trainable
        self.setup()

        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)

    def setup(self):
        l1=layers.conv2d(inputs=1-self.data,num_outputs=32,kernel_size=[3,3],stride=[2,2])
        l1_bis=layers.conv2d(inputs=l1,num_outputs=64,kernel_size=[3,3],stride=[2,2])
        l2 = layers.conv2d(inputs=l1_bis, num_outputs=64, kernel_size=[3, 3], stride=[2, 2])

        l3=layers.conv2d(inputs=l2, num_outputs=128, kernel_size=[3, 3], stride=[1, 1]) # layer 3 saves our ass 3 pq pas 8 ou 16 ou plus ?
        self.rpn_cls_score=layers.conv2d(inputs=l3, num_outputs=len(anchor_scales)*2, kernel_size=[1, 1], stride=[1, 1],padding='VALID',activation_fn=None)
        self.rpn_bbox_pred=layers.conv2d(inputs=l3, num_outputs=len(anchor_scales)*4, kernel_size=[1, 1], stride=[1, 1],padding='VALID',activation_fn=None)

        self.rpn_cls_prob = tf.nn.softmax(self.rpn_cls_score)

        if self.state=="TRAIN":
            self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights, debug_info = tf.py_func(anchor_target_layer, [self.rpn_cls_score, self.gt_boxes, self.im_info, self.data, _feat_stride, anchor_scales],
                       [tf.int32, tf.float32, tf.float32, tf.float32, tf.float32])

            self.debug_info = debug_info

        self.rpn_rois=tf.reshape(tf.py_func(proposal_layer, [self.rpn_cls_prob, self.rpn_bbox_pred, self.im_info, self.state, _feat_stride, anchor_scales],
                              [tf.float32]), [-1, 5], name='rpn_rois')

        if self.state=="TRAIN":
            rois, self.labels, self.bbox_targets, self.bbox_inside_weights, self.bbox_outside_weights = tf.py_func(proposal_target_layer,
                                                                                               [self.rpn_rois, self.gt_boxes, n_classes],
                                                                                               [tf.float32, tf.int32,
                                                                                                tf.float32, tf.float32, tf.float32])
            rois = tf.reshape(rois, [-1, 5], name='rois')

        l2_swapped = tf.transpose(l2, perm=[0, 3, 1, 2])
        output_shape_tf = tf.constant((7, 7))

        if self.state=="TRAIN":
            #rois_pooled = roi_pool_op.roi_pool(l2, rois,7,7,1/16.0,name='pool_5')[0]
            new_rois, = tf.py_func(adapt_rois, [rois], [tf.int32])
        else:
            #rois_pooled = roi_pool_op.roi_pool(l2, self.rpn_rois,7,7,1/16.0,name='pool_5')[0]
            # 1. Adapt rois
            new_rois, = tf.py_func(adapt_rois, [self.rpn_rois], [tf.int32])

        rois_pooled_before, argmax = roi_pooling_op_2(l2_swapped, new_rois, output_shape_tf)
        rois_pooled_transposed = tf.transpose(rois_pooled_before, perm=[0, 2, 1, 3, 4])
        rois_pooled = tf.reshape(rois_pooled_transposed, [-1, 64, 7, 7])  # Be careful ! The final depth is 64
        # output : [batch_size, 7, 7, features_depth]

        l5 = layers.flatten(rois_pooled)
        fc6=layers.fully_connected(l5,128)
        fc6=layers.dropout(fc6,is_training=self.state=="TRAIN")
        fc7=layers.fully_connected(fc6,128)
        fc7=layers.dropout(fc7,is_training=self.state=="TRAIN")
        self.logits=layers.fully_connected(fc7,n_classes,activation_fn=None)
        self.cls_prob=layers.softmax(self.logits)

        self.bbox_pred = layers.fully_connected(fc7, 4*n_classes, activation_fn=None,scope='bbox_pred')

    def compute_loss_and_summaries(self):
        with tf.name_scope('rpn'):
            with tf.name_scope('cls'):
                # self.net.get_output('rpn_cls_score_reshape')
                rpn_cls_score = tf.reshape(self.rpn_cls_score, [-1, 2])

                # tf.logical_and(tf.not_equal(rpn_label,-1),tf.not_equal(rpn_label,0))
                rpn_cls_score_selected = tf.reshape(tf.gather(rpn_cls_score, tf.where(tf.not_equal(self.rpn_labels, -1))),
                                                    [-1, 2])
                rpn_labels_selected = tf.reshape(tf.gather(self.rpn_labels, tf.where(tf.not_equal(self.rpn_labels, -1))), [-1])
                # tf.summary.histogram('cls_score',rpn_cls_score_selected)
                rpn_fg_acc = tf.reduce_mean(
                    tf.cast(tf.equal(tf.greater_equal(rpn_cls_score_selected[:, 0], 0.5), tf.equal(rpn_labels_selected, 0)),
                            tf.float32))
                tf.summary.scalar('fg_acc', rpn_fg_acc)
                rpn_cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score_selected, labels=rpn_labels_selected))
                tf.summary.scalar('loss', rpn_cross_entropy)
            with tf.name_scope('bbox'):
                rpn_smooth_l1 = smooth_l1(3.0, self.rpn_bbox_pred, self.rpn_bbox_targets, self.rpn_bbox_inside_weights,
                                          self.rpn_bbox_outside_weights)
                rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))
                tf.summary.scalar('loss', rpn_loss_box)
            rpn_loss = rpn_cross_entropy + rpn_loss_box
            tf.summary.scalar('loss', rpn_loss)

        # R-CNN
        with tf.name_scope('rcnn'):
            with tf.name_scope('cls'):
                labels = tf.reshape(self.labels, [-1])
                cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=labels))
                tf.summary.scalar('loss', cross_entropy)
                bg_acc = tf.reduce_mean(
                    tf.cast(tf.equal(tf.greater_equal(self.cls_prob[:, 0], 0.5), tf.equal(labels, 0)), tf.float32))
                cl1_acc = tf.reduce_mean(
                    tf.cast(tf.equal(tf.greater_equal(self.cls_prob[:, 1], 0.5), tf.equal(labels, 1)), tf.float32))
                tf.summary.scalar('bg_acc', bg_acc)
                tf.summary.scalar('cl1_acc', cl1_acc)
            with tf.name_scope('bbox'):
                loss_box = smooth_l1(1.0, self.bbox_pred, self.bbox_targets, self.bbox_inside_weights, self.bbox_outside_weights)
                loss_box = tf.reduce_mean(tf.reduce_sum(loss_box, reduction_indices=[1]))
                tf.summary.scalar('loss', loss_box)
            rcnn_loss = cross_entropy + loss_box
            tf.summary.scalar('loss', rcnn_loss)

        with tf.name_scope('total_loss'):
            loss = rpn_loss + rcnn_loss
            tf.summary.scalar('loss', loss)

        # rpn_fg_acc, rpn_cross_entropy, rpn_loss_box

        self.train_op = tf.train.AdamOptimizer(5e-4).minimize(loss)
        self.merge = tf.summary.merge_all()
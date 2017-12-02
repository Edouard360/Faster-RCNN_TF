import tensorflow as tf
from rpn_msr.anchor_target_layer_tf import anchor_target_layer
from rpn_msr.proposal_layer_tf import proposal_layer
from rpn_msr.proposal_target_layer_tf import proposal_target_layer
from networks.network import Network
import roi_pooling_layer.roi_pooling_op as roi_pool_op

import tensorflow.contrib.layers as layers
# layers.conv2d #
#define

n_classes =  3
_feat_stride = [16,]
anchor_scales = [40]#[8, 16, 32]

class VGGnet_trainoverfit(Network):
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
        l1=layers.conv2d(inputs=self.data,num_outputs=32,kernel_size=[3,3],stride=[4,4])
        l2=layers.conv2d(inputs=l1,num_outputs=64,kernel_size=[3,3],stride=[4,4])

        l3=layers.conv2d(inputs=l2, num_outputs=128, kernel_size=[3, 3], stride=[1, 1])
        self.rpn_cls_score=layers.conv2d(inputs=l3, num_outputs=len(anchor_scales)*2, kernel_size=[1, 1], stride=[1, 1],padding='VALID',activation_fn=None)
        self.rpn_bbox_pred=layers.conv2d(inputs=l3, num_outputs=len(anchor_scales)*4, kernel_size=[1, 1], stride=[1, 1],padding='VALID',activation_fn=None)
        # The channel (4) is at the end

        self.rpn_cls_prob = tf.nn.softmax(self.rpn_cls_score)

        if self.state=="TRAIN":
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, debug_info = tf.py_func(anchor_target_layer, [self.rpn_cls_score, self.gt_boxes, self.im_info, self.data, _feat_stride, anchor_scales],
                       [tf.int32, tf.float32, tf.float32, tf.float32, tf.float32])
            self.rpn_bbox_targets = rpn_bbox_targets
            self.rpn_bbox_inside_weights = rpn_bbox_inside_weights
            self.rpn_bbox_outside_weights = rpn_bbox_outside_weights
            self.rpn_labels = rpn_labels
            self.debug_info = debug_info

        self.rpn_rois=tf.reshape(tf.py_func(proposal_layer, [self.rpn_cls_prob, self.rpn_bbox_pred, self.im_info, self.state, _feat_stride, anchor_scales],
                              [tf.float32]), [-1, 5], name='rpn_rois')

        if self.state=="TRAIN":
            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(proposal_target_layer,
                                                                                               [self.rpn_rois, self.gt_boxes, n_classes],
                                                                                               [tf.float32, tf.int32,
                                                                                                tf.float32, tf.float32, tf.float32])
            rois = tf.reshape(rois, [-1, 5], name='rois')
            self.labels = labels
            self.bbox_targets = bbox_targets
            self.bbox_inside_weights = bbox_inside_weights
            self.bbox_outside_weights = bbox_outside_weights


        if self.state=="TRAIN":
            rois_pooled = roi_pool_op.roi_pool(l2, rois,7,7,1/16.0,name='pool_5')[0]
        else:
            rois_pooled = roi_pool_op.roi_pool(l2, self.rpn_rois,7,7,1/16.0,name='pool_5')[0]
        # output : [batch_size, 7, 7, features_depth]

        l5 = layers.flatten(rois_pooled)
        fc6=layers.fully_connected(l5,128)
        fc6=layers.dropout(fc6,is_training=self.state=="TRAIN")
        fc7=layers.fully_connected(fc6,128)
        fc7=layers.dropout(fc7,is_training=self.state=="TRAIN")
        self.logits=layers.fully_connected(fc7,n_classes,activation_fn=None)
        self.cls_prob=layers.softmax(self.logits)

        self.bbox_pred = layers.fully_connected(fc7, 4*n_classes, activation_fn=None,scope='bbox_pred')
        '''
        (self.feed('data') # parentesis is to avoid backslash
             .conv(3, 3, 16, 4, 4, name='conv1_1')
             .conv(3, 3, 32, 4, 4, name='conv5_3'))
        #========= RPN ============
        (self.feed('conv5_3')
             .conv(3,3,128,1,1,name='rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score'))
            # *3 for the ratio - * 2 object or not in the background TODO reput it

        (self.feed('rpn_cls_score','gt_boxes','im_info','data') # Only the shape of rpn_cls_score is used
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' ))

        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred'))
        # *3 for the ratio - *4 for the bbox prediction TODO reput it

        #========= RoI Proposal ============
        (self.feed('rpn_cls_score')
             .reshape_layer(2,name = 'rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*2,name = 'rpn_cls_prob_reshape'))
        #*3 for the ration TODO reput it

        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'rpn_rois'))

        (self.feed('rpn_rois','gt_boxes')
             .proposal_target_layer(n_classes,name = 'roi-data'))


        #========= RCNN ============
        out = (self.feed('conv5_3', 'roi-data')
             .roi_pool(7, 7, 1.0/16, name='pool_5'))
        # TODO: 16 is spatial scale = feature stride ????
        with tf.variable_scope('FC6'):
            out = (out.fc(128, name='fc6')
                 .dropout(0.5, name='drop6'))
        with tf.variable_scope('FC7'):
            out = (out.fc(128, name='fc7')
                 .dropout(0.5, name='drop7'))

        (out.fc(n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

        (self.feed('drop7')
             .fc(n_classes*4, relu=False, name='bbox_pred'))
        '''



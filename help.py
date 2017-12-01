import matplotlib.pyplot as plt
# im_file = os.path.join(cfg.DATA_DIR, 'demo', "02.jpg")
# im = cv2.imread(im_file)
# dets = np.array([gt_boxes[:, :4], 1])
# cls="DICK"
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(12, 12))
ax.imshow(im, aspect='equal')

# LOOK in config (experiments)

# Proposal target layer for debugging the bounding boxes

# --weights /Users/edouardm/Desktop/Faster-RCNN_TF/output/faster_rcnn_end2end/voc_2008_trainval/VGGnet_fast_rcnn_iter_800.ckpt



###### Trash.py

#global_step = tf.Variable(0, trainable=False)
# cfg.TRAIN.LEARNING_RATE
# lr = tf.train.exponential_decay(0.1, global_step,
#                                 cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
# momentum = cfg.TRAIN.MOMENTUM
# train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step)
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

## In anchor target layer py
# import ipdb;

# fig, ax = plt.subplots(figsize=(12, 12))
# ax.imshow(data[0], aspect='equal')
# vis_detections(None,"1",np.concatenate((anchors[:3,:],np.ones((3,1),dtype=np.int32)),axis=1),ax)

# ANCHORS 1 - SIGMA
# vis_detections(None, "2", np.concatenate((anchors[40000:40009, :], np.ones((9, 1),dtype=np.int32)), axis=1), ax)
# vis_detections(None,"3",np.concatenate((anchors[60000:60003,:],np.ones((3,1),dtype=np.int32)),axis=1),ax)
# vis_detections(None, "2", np.concatenate((anchors[16000:16009, :], np.ones((10, 1),dtype=np.int32)), axis=1), ax)

# ANCHORS 2 - SIGMA
# vis_detections(None,"3",np.concatenate((anchors[3000:3003,:],np.ones((3,1),dtype=np.int32)),axis=1),ax)
# vis_detections(None,"3",np.concatenate((anchors[8000:8003,:],np.ones((3,1),dtype=np.int32)),axis=1),ax)

# ANCHORS 3 - SIGMA
# vis_detections(None,"3",np.concatenate((anchors[3000:3003,:],np.ones((3,1),dtype=np.int32)),axis=1),ax)
# vis_detections(None,"3",np.concatenate((anchors[8000:8003,:],np.ones((3,1),dtype=np.int32)),axis=1),ax)

# vis_detections(None, "3", np.concatenate((anchors[3000:3015:5, :], np.ones((3, 1), dtype=np.int32)), axis=1), ax)
# fig.savefig("anchors3")
# ipdb.set_trace()

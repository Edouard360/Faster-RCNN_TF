python2.7 ./tools/demo.py --cpu \
--model /Users/edouardm/Desktop/Faster-RCNN_TF/VGGnet_fast_rcnn_iter_70000.ckpt

# To start with 0 weights initially !
python tools/train_net.py \
--imdb voc_2007_trainval \
--network VGGnet_trainoverfit \
--cfg /Users/edouardm/Desktop/Faster-RCNN_TF/experiments/cfgs/faster_rcnn_end2end.yml

# Otherwise add the line
--weights /Users/edouardm/Desktop/Faster-RCNN_TF/VGGnet_fast_rcnn_iter_70000.ckpt \

# Now things get serious
python2.7 ./tools/demo.py --cpu \
--model /Users/edouardm/Desktop/Faster-RCNN_TF/output/faster_rcnn_end2end/voc_2007_trainval/VGGnet_fast_rcnn_iter_2.ckpt
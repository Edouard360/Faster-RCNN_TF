#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import argparse
import pprint
import sys

from customNet import CustomNet
from datasets.my_dataset import my_dataset
from main_train import train_net


from roi_data_layer.roidb import prepare_roidb

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')

    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)

    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    imdb = my_dataset()
    print 'Loaded dataset'
    prepare_roidb(imdb)
    roidb = imdb.roidb
    network = CustomNet()

    train_net(network, imdb, roidb,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)

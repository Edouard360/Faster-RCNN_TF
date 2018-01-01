# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
from config import TEXT_CLASSES

class my_dataset(imdb):
    def __init__(self):
        imdb.__init__(self, '')
        self._classes = TEXT_CLASSES

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        with open('data/ImageSets/trainval.txt') as f:
            image_index = [x.strip() for x in f.readlines()]
        self._image_index = image_index
        self._roidb_handler = self.gt_roidb

    def image_path_at(self, i):
        return 'data/JPEGImages/'+self._image_index[i]+'.jpg'

    def gt_roidb(self):
        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        return gt_roidb

    def rpn_roidb(self):
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        return roidb


    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = 'data/Annotations/'+index+ '.xml'
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text] #TODO : this prevented sigma Sigma difference - removed .lower().strip()
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}


if __name__ == '__main__':
    from datasets.my_dataset import my_dataset
    d = my_dataset()
    res = d.roidb
    from IPython import embed; embed()

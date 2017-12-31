import argparse
import os

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from bbox import xml_bboxs, filter_bboxs
from write_xml import write_xml

from py_cpu_nms import py_cpu_nms

# This is a CONSTANT of all the SYMBOLS. Can be simplified.
from constants import SYMBOLS_CLASSES


def generate_symbols_and_xml(n_images=1, n_samples_per_image=10, max_overlap=0.15, filepath='', image_folder='',
                             annotation_folder=''):
    width = 600
    height = 600
    dpi = 192  # Change that according to your computer
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width / float(dpi), height / float(dpi)), dpi=dpi)
    ax.set_axis_off()  # clears the axis

    for i in range(n_images):
        plt.clf()
        filename = '{0:03}'.format(i)

        position = np.random.uniform(low=0.15, high=0.65, size=(n_samples_per_image, 2))

        taille = 60 + np.random.normal(0, 5, size=n_samples_per_image)
        symbols = np.random.choice(SYMBOLS_CLASSES, size=n_samples_per_image)
        texts = [fig.text(x, y, symbol, size=s) for (x, y), s, symbol in zip(position, taille, symbols)]
        fig.savefig(filepath + image_folder + filename + '.jpg', dpi=dpi)
        bboxs = [text.get_window_extent() for text in texts]

        bboxs_bounds = xml_bboxs(bboxs, width, height)

        dets = bboxs_bounds.copy()
        dets[:, [1, 3]] = dets[:, [3, 1]]
        valid_indices = filter_bboxs(dets, width, height)
        dets = dets[valid_indices]
        dets = np.concatenate((dets, np.arange(len(dets), 0, -1).reshape(-1, 1)), axis=1)
        indices_in_valid = np.array(py_cpu_nms(dets, max_overlap))

        indices = valid_indices[indices_in_valid]
        print 'Nb. of kept indices : %d/%d' % (len(indices), n_samples_per_image)

        # Once we have decided which indices to keep, we clear the plot
        # And overwrite, to plot only the non-overlapping symbols
        plt.clf()
        for (x, y), s, symbol in zip(position[indices], taille[indices], symbols[indices]):
            fig.text(x, y, symbol, size=s)
        fig.savefig(filepath + image_folder + filename + '.jpg', dpi=dpi)

        if annotation_folder is not None:
            write_xml(symbols[indices], bboxs_bounds[indices], width, height,
                      filename=filepath + annotation_folder + filename + '.xml')


def generate_symbols_only(**kargs):
    generate_symbols_and_xml(annotation_folder=None, **kargs)


def generate_symbols_and_xml_to_voc2008(**kargs):
    generate_symbols_and_xml(filepath=os.path.dirname(os.path.abspath(__file__)) + '/../data/VOCdevkit2008/VOC2008/',
                             image_folder='JPEGImages/',
                             annotation_folder='Annotations/',
                             **kargs)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate Images')

    parser.add_argument('--n_images', help='number of images to generate',
                        default=1, type=int)
    parser.add_argument('--n_samples_per_image', help='max number of symbols we will plot in am image',
                        default=10, type=int)
    parser.add_argument('--max_overlap', help='max_overlap allowed between images',
                        default=0.15, type=float)
    parser.add_argument('--annotation_folder', help='where annotations should go',
                        default='', type=str)
    parser.add_argument('--image_folder', help='where images should go',
                        default='', type=str)
    parser.add_argument('--filepath', help='you can indicate a common filepath '
                                           'to image_folder and annotation_folder (prefix) ',
                        default='', type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    generate_symbols_and_xml(**vars(args))

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from bbox import coords_bboxs, xml_bboxs, filter_bboxs
from write_xml import write_xml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))

from nms.py_cpu_nms import py_cpu_nms

width = 600
height = 600

dpi = 192  # Change that according to your computer
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width / float(dpi), height / float(dpi)), dpi=dpi)
ax.set_axis_off()  # clears the axis
n_images = 10

n_samples_per_image = 12

filepath = os.path.dirname(os.path.abspath(__file__)) + '/../data/VOCdevkit2008/VOC2008/'

with open(filepath + "ImageSets/Main/trainval.txt", "a") as myfile:
    for i in range(n_images):
        plt.clf()
        filename = '{0:03}'.format(i + 10)
        myfile.write('\n' + filename)

        symbols_classes = [r'$\longrightarrow$', r'$\sigma$', r'$\alpha$', r'$\gamma$',
                           r'$\int$']  # r'$\int$' ,]#['$\Sigma$','$\sigma$']#,,'A','B','a','b']
        position = np.random.uniform(low=0.15, high=0.65, size=(n_samples_per_image, 2))
        # x = np.linspace(0.3, 0.6, 2)
        # y = np.linspace(0.3, 0.6, 2)
        # xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
        # k=0
        # for x_i in range(2):
        #     for y_j in range(2):
        #         position[k] = [xv[x_i,y_j], yv[x_i,y_j]]
        #         k+=1
        taille = 60 + np.random.normal(0, 5, size=n_samples_per_image)
        symbols = np.random.choice(symbols_classes, size=n_samples_per_image)
        texts = [fig.text(x, y, symbol, size=s) for (x, y), s, symbol in zip(position, taille, symbols)]
        fig.savefig(filepath + 'JPEGImages/' + filename + '.jpg', dpi=dpi)
        bboxs = [text.get_window_extent() for text in texts]

        bboxs_coords = coords_bboxs(bboxs, width, height)

        # [plot_bbox(ax,bbox_coord) for bbox_coord in bboxs_coords]
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        # fig.savefig('symbol_with_bbox.jpg', dpi = dpi)
        # plt.show()

        bboxs_bounds = xml_bboxs(bboxs, width, height)

        dets = bboxs_bounds.copy()
        dets[:, [1, 3]] = dets[:, [3, 1]]
        valid_indices = filter_bboxs(dets, width, height)
        dets = dets[valid_indices]
        dets = np.concatenate((dets, np.arange(len(dets), 0, -1).reshape(-1, 1)), axis=1)
        indices_in_valid = np.array(py_cpu_nms(dets, 0.15))

        indices = valid_indices[indices_in_valid]
        print 'Nb. of kept indices : ', len(indices)

        # TODO: unfortunately we rewrite brutally here...
        plt.clf()
        texts = [fig.text(x, y, symbol, size=s) for (x, y), s, symbol in
                 zip(position[indices], taille[indices], symbols[indices])]
        fig.savefig(filepath + 'JPEGImages/' + filename + '.jpg', dpi=dpi)

        write_xml(symbols[indices], bboxs_bounds[indices], width, height,
                  filename=filepath + 'Annotations/' + filename + '.xml')

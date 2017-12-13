import matplotlib.pyplot as plt
import numpy as np
from write_xml import write_xml
from bbox import plot_bbox,coords_bboxs,xml_bboxs

width = 600
height = 600

dpi = 192 # Change that according to your computer
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(width/float(dpi), height/float(dpi)),dpi=dpi)
ax.set_axis_off() #clears the axis
n_images = 20
n_samples_per_image = 3

for i in range(n_images):
    plt.clf()
    filename = '{0:03}'.format(i)
    symbols_classes = ['$\Sigma$','$\sigma$','$\gamma$','A','B','a','b']
    position = np.random.uniform(low=0.25,high=0.75,size=(n_samples_per_image,2))
    taille = 40 + np.random.normal(0,5,size=n_samples_per_image)
    symbols = np.random.choice(symbols_classes,size=n_samples_per_image)
    texts=[fig.text(x,y, symbol, size = s) for (x,y), s, symbol in zip(position,taille,symbols)]
    fig.savefig(filename+'.jpg', dpi = dpi)
    bboxs = [text.get_window_extent() for text in texts]

    #bboxs_coords = coords_bboxs(bboxs,width,height)
    #[plot_bbox(ax,bbox_coord) for bbox_coord in bboxs_coords]
    #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    #fig.savefig('symbol_with_bbox.jpg', dpi = dpi)
    #plt.show()

    bboxs_bounds = xml_bboxs(bboxs, width, height)

    write_xml(symbols, bboxs_bounds, width, height, filename=filename+'.xml')
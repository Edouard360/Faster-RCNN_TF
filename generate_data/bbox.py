import matplotlib.pyplot as plt

def plot_bbox(ax,bbox):
    """Plot bounding-box on matplotlib ax"""
    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                  bbox[2] - bbox[0],
                  bbox[3] - bbox[1], fill=False,
                  edgecolor='red', linewidth=2)
    )

def xml_bboxs(bboxs, width, height):
    # From matplotlib to xml - Pascal format y-axis is reversed
    bboxs_bounds = []
    for bbox in bboxs:
        bounds = bbox.bounds
        bboxs_bounds+=[[bounds[0],height-bounds[1],bounds[0]+bounds[2],height-(bounds[1]+bounds[3])]]
    return bboxs_bounds

def coords_bboxs(bboxs,width,height):
    # From matplotlib to [0,1] coords
    bboxs_coords = []
    for bbox in bboxs:
        bounds = bbox.bounds
        bboxs_coords+=[[bounds[0]/width,bounds[1]/height,(bounds[0]+bounds[2])/width,(bounds[1]+bounds[3])/height]]
    return bboxs_coords
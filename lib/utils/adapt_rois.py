import numpy as np

def adapt_rois(rois):
    rois = rois[:, 1:]
    # rois[:,0]=0
    # rois[:,1] = 0
    # rois[:,2] = 1000
    # rois[:,3] = 1000

    rois = rois/8.0 #TODO: add to the list of things to be careful about
    rois[:,2]=rois[:,2]-rois[:,0] # -20 # margin be careful ?
    rois[:, 3] = rois[:, 3] - rois[:, 1] #-20
    #print("rois mean width ",rois[:, 2].mean())
    test = np.array([rois], dtype=np.int32)
    #print("test",test.shape)
    return np.array([rois],dtype=np.int32) # additional [] to specify it is the first image

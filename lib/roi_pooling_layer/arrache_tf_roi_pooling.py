import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

'''
BELOW: example
'''
# Def custom square function using np.square instead of tf.square:
def mysquare(x, name=None):
    with ops.op_scope([x], name, "Mysquare") as name:
        tf.RegisterGradient("mysquare")(_MySquareGrad)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": "mysquare"}):
            sqr_x = tf.py_func(np.vectorize(lambda x:np.power(x,3,dtype=np.float32)),
                            [x],
                            [tf.float32],
                            name=name,
                            stateful=True)  # <-- here's the call to the gradient
        return sqr_x[0]


# Actual gradient:
def _MySquareGrad(op, grad):
    x = op.inputs[0]
    return grad * 3*x**2  # add a "small" error just to see the difference:


with tf.Session() as sess:
    x = tf.constant([2.])#, 2.
    y = mysquare(x)
    sess.run(tf.global_variables_initializer())

    print(x.eval(), y.eval(), tf.gradients(y, x)[0].eval())

'''
ATTEMPT: roi pooling # .self prevents cooperation between gradients and inference...
'''

def forward(x, name=None):
    with ops.op_scope([x], name, "Mysquare") as name:
        tf.RegisterGradient("roi")(backward)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": "roi"}):
            sqr_x = tf.py_func(np.vectorize(lambda x:np.power(x,3,dtype=np.float32)),
                            [x,],
                            [tf.float32,tf.float32],
                            name=name,
                            stateful=True)  # <-- here's the call to the gradient
        return sqr_x[0]

def forward(inputs):
    pass

def backward(op, grad):
    spatial_scale=1/16.0
    outh=7
    outw=7
    bottom_rois = inputs[1]
    channels, height, width = self._bottom_data_shape[1:]
    n_rois = bottom_rois.shape[0]
    bottom_delta = np.zeros(self._bottom_data_shape, np.float32)

    for i_roi in six.moves.range(n_rois):
        idx, xmin, ymin, xmax, ymax = bottom_rois[i_roi]
        idx = int(idx)
        xmin = int(round(xmin * self.spatial_scale))
        xmax = int(round(xmax * self.spatial_scale))
        ymin = int(round(ymin * self.spatial_scale))
        ymax = int(round(ymax * self.spatial_scale))
        roi_width = max(xmax - xmin + 1, 1)
        roi_height = max(ymax - ymin + 1, 1)

        strideh = float(roi_height) / float(self.outh)
        stridew = float(roi_width) / float(self.outw)

        # iterate all the w, h (from feature map) that fall into this ROIs
        for w in six.moves.range(xmin, xmax + 1):
            for h in six.moves.range(ymin, ymax + 1):
                phstart = int(np.floor(float(h - ymin) / strideh))
                phend = int(np.ceil(float(h - ymin + 1) / strideh))
                pwstart = int(np.floor(float(w - xmin) / stridew))
                pwend = int(np.ceil(float(w - xmin + 1) / stridew))

                phstart = min(max(phstart, 0), self.outh)
                phend = min(max(phend, 0), self.outh)
                pwstart = min(max(pwstart, 0), self.outw)
                pwend = min(max(pwend, 0), self.outw)

                for ph in six.moves.range(phstart, phend):
                    for pw in six.moves.range(pwstart, pwend):
                        max_idx_tmp = self.argmax_data[i_roi, :, ph, pw] #TODO ? impossible to replace as such
                        for c in six.moves.range(channels):
                            if max_idx_tmp[c] == (h * width + w):
                                bottom_delta[idx, c, h, w] += \
                                    grad[0][i_roi, c, ph, pw]
    return bottom_delta, None

def roi_pooling2d(x,y, name=None):
    with ops.op_scope([x], name, "Mysquare") as name:
        tf.RegisterGradient("roi")(_MySquareGrad)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": "roi"}):
            sqr_x = tf.py_func(forward,
                            [x,y],
                            [tf.float32],
                            name=name,
                            stateful=True)  # <-- here's the call to the gradient
        return sqr_x[0]

'''
def forward_cpu(self, inputs):
    self.retain_inputs((1,))
    self._bottom_data_shape = inputs[0].shape

    bottom_data, bottom_rois = inputs
    channels, height, width = bottom_data.shape[1:]
    n_rois = bottom_rois.shape[0]
    # `np.zeros` needs to be used because the arrays can be
    # returned without having some of its values updated.
    top_data = np.zeros((n_rois, channels, self.outh, self.outw),
                           dtype=np.float32)
    self.argmax_data = np.zeros(top_data.shape, np.int32)

    for i_roi in six.moves.range(n_rois):
        idx, xmin, ymin, xmax, ymax = bottom_rois[i_roi]
        xmin = int(round(xmin * self.spatial_scale))
        xmax = int(round(xmax * self.spatial_scale))
        ymin = int(round(ymin * self.spatial_scale))
        ymax = int(round(ymax * self.spatial_scale))
        roi_width = max(xmax - xmin + 1, 1)
        roi_height = max(ymax - ymin + 1, 1)
        strideh = 1. * roi_height / self.outh
        stridew = 1. * roi_width / self.outw

        for outh in six.moves.range(self.outh):
            sliceh, lenh = _roi_pooling_slice(
                outh, strideh, height, ymin)
            if sliceh.stop <= sliceh.start:
                continue
            for outw in six.moves.range(self.outw):
                slicew, lenw = _roi_pooling_slice(
                    outw, stridew, width, xmin)
                if slicew.stop <= slicew.start:
                    continue
                roi_data = bottom_data[int(idx), :, sliceh, slicew] \
                    .reshape(channels, -1)
                top_data[i_roi, :, outh, outw] = \
                    np.max(roi_data, axis=1)

                # get the max idx respect to feature_maps coordinates
                max_idx_slice = np.unravel_index(
                    np.argmax(roi_data, axis=1), (lenh, lenw))
                max_idx_slice_h = max_idx_slice[0] + sliceh.start
                max_idx_slice_w = max_idx_slice[1] + slicew.start
                max_idx_slice = max_idx_slice_h * width + max_idx_slice_w
                self.argmax_data[i_roi, :, outh, outw] = max_idx_slice
    return top_data,

def backward_cpu(self, inputs, gy):
    bottom_rois = inputs[1]
    channels, height, width = self._bottom_data_shape[1:]
    n_rois = bottom_rois.shape[0]
    bottom_delta = np.zeros(self._bottom_data_shape, np.float32)

    for i_roi in six.moves.range(n_rois):
        idx, xmin, ymin, xmax, ymax = bottom_rois[i_roi]
        idx = int(idx)
        xmin = int(round(xmin * self.spatial_scale))
        xmax = int(round(xmax * self.spatial_scale))
        ymin = int(round(ymin * self.spatial_scale))
        ymax = int(round(ymax * self.spatial_scale))
        roi_width = max(xmax - xmin + 1, 1)
        roi_height = max(ymax - ymin + 1, 1)

        strideh = float(roi_height) / float(self.outh)
        stridew = float(roi_width) / float(self.outw)

        # iterate all the w, h (from feature map) that fall into this ROIs
        for w in six.moves.range(xmin, xmax + 1):
            for h in six.moves.range(ymin, ymax + 1):
                phstart = int(np.floor(float(h - ymin) / strideh))
                phend = int(np.ceil(float(h - ymin + 1) / strideh))
                pwstart = int(np.floor(float(w - xmin) / stridew))
                pwend = int(np.ceil(float(w - xmin + 1) / stridew))

                phstart = min(max(phstart, 0), self.outh)
                phend = min(max(phend, 0), self.outh)
                pwstart = min(max(pwstart, 0), self.outw)
                pwend = min(max(pwend, 0), self.outw)

                for ph in six.moves.range(phstart, phend):
                    for pw in six.moves.range(pwstart, pwend):
                        max_idx_tmp = self.argmax_data[i_roi, :, ph, pw]
                        for c in six.moves.range(channels):
                            if max_idx_tmp[c] == (h * width + w):
                                bottom_delta[idx, c, h, w] += \
                                    gy[0][i_roi, c, ph, pw]
    return bottom_delta, None

'''
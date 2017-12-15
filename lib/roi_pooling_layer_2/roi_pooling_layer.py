import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

wd = os.path.dirname(__file__)#os.path.abspath(os.path.dirname(__file__)) # get the working directory

roi_pooling_module_2 = tf.load_op_library(wd+"/roi_pooling.so")
roi_pooling_op_2 = roi_pooling_module_2.roi_pooling2
roi_pooling_module_2_grad = tf.load_op_library(wd+"/roi_pooling_op_grad.so")
roi_pooling_grad_op_2 = roi_pooling_module_2_grad.roi_pooling2_grad

# Here we register our gradient op as the gradient function for our ROI pooling op.
@ops.RegisterGradient("RoiPooling2")
def _roi_pooling_grad(op, grad0, grad1):
    # The input gradients are the gradients with respect to the outputs of the pooling layer
    input_grad = grad0

    # We need the argmax data to compute the gradient connections
    argmax = op.outputs[1]

    # Grab the shape of the inputs to the ROI pooling layer
    input_shape = array_ops.shape(op.inputs[0])

    # Compute the gradient -> roi_pooling_op_grad is not defined ...
    backprop_grad = roi_pooling_grad_op_2(input_grad, argmax, input_shape)

    # Return the gradient for the feature map, but not for the other inputs
    return [backprop_grad, None, None]
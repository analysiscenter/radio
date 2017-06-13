"""Helper functions for creating layers """

import sys
import tensorflow as tf


def conv3d_bnorm_activation(scope, inputs, training, activation=tf.nn.relu,
                            kernel=[7, 7, 7], channels=1):
    """
    form conv3d -> batch norm -> relu
    Args:
        inputs: input layer, 5d-tensor: (None, depth, width, height, input_channels)
        training: boolean tf placeholder; if True then bnorm is used in training mode
        activation: nonlinearity to use after bnorm
        kernel: conv3d filter size
        scope: scope to create ops in
    return:
        output, 5d-tensor: (None, depth, width, height, channels)
    """
    with tf.variable_scope(scope):
        conv = tf.layers.conv3d(inputs, filters=channels, kernel_size=kernel,
                                padding='same', name='convolution')
        normed = tf.layers.batch_normalization(
            conv, training=training, name='batch-norm')
        if activation is None:
            return normed
        else:
            return activation(normed)


def deconv3d_bnorm_activation(scope, inputs, training, activation=tf.nn.relu,
                              kernel=[5, 5, 5], channels=1):
    """
    form upsampling deconv3d -> batch norm -> relu

    *Note: this is doubling deconv-layer
        that is, the stride is always 2 in all spatial dims

    Args:
        inputs: input layer, 5d-tensor: (None, depth, width, height, input_channels)
        training: boolean tf placeholder; if True then bnorm is used in training mode
        activation: nonlinearity to use after bnorm
        kernel: conv3d filter size
        scope: scope to create ops in
    return:
        output, 5d-tensor: (None, 2 * depth, 2 * width, 2 * height, channels)
    """
    with tf.variable_scope(scope):
        # set output shape (double input shape in 3 spatial dims)
        input_shape = inputs.shape
        inp_shape = tf.shape(inputs)
        output_shape = ([inp_shape[0]] +
                        [2 * int(input_shape[i]) for i in range(1, len(input_shape) - 1)] +
                        [channels])
        # set strides for doubling
        strides = [1, 2, 2, 2, 1]

        # create filter variable
        ftr = tf.Variable(tf.random_normal(shape=list(
            kernel) + [channels] + [int(input_shape[-1])]))

        # apply deconvolution
        deconv = tf.nn.conv3d_transpose(inputs, ftr, output_shape, strides,
                                        padding='SAME', name='deconvolution')
        # add bias
        bias = tf.Variable(tf.random_normal([channels]))
        deconv = tf.nn.bias_add(deconv, bias)

        normed = tf.layers.batch_normalization(
            deconv, training=training, name='batch-norm')
        if activation is None:
            return normed
        else:
            return activation(normed)

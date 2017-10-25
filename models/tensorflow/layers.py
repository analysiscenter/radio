"""Helper functions for creating layers """

import tensorflow as tf


def get_initializer(initializer):
    """ Get tensorflow initializer.

    Returns normal initializer or xavier initializer.

    Args:
    - kernel_init: str, can be 'xavier' or 'normal' initializer for kernel;

    Returns:
    - initializer;
    """
    kenel_init_fn = None
    if kernel_init == 'xavier':
        kernel_init_fn = tf.contrib.layers.xavier_initializer
    elif kernel_init == 'normal':
        kernel_init_fn = tf.random_normal_initializer
    else:
        raise ValueError("Argument kernel_init must have 'str' type " +
                         "and be 'xavier' or 'normal'")
    return kenel_init_fn


def selu(x):
    """
    selu activation function
    Args:
        input tensor x
    Return:
        selu(x)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def conv3d(self, input_tensor, filters, kernel_size, name,
           strides=(1, 1, 1), padding='same', activation=tf.nn.relu,
           use_bias=True, kernel_init='xavier', is_training=True):
    """ Apply 3D convolution operation to input tensor.

    Args:
    - input_tensor: tf.Variable, input tensor;
    - filters: int, number of filters in the ouput tensor;
    - kernel_size: tuple(int, int, int), size of kernel
      of 3D convolution operation along 3 dimensions;
    - name: str, name of the layer that will be used as an argument of tf.variable_scope;
    - strides: tuple(int, int, int), size of strides along 3 dimensions;
    - padding: str, padding mode, can be 'same' or 'valid';
    - activation: tensorflow activation function that will be applied to
    output tensor;
    - use_bias: bool, whether use bias or not;
    - kernel_init: str, can be 'xavier' or 'normal' initializer for kernel;

    Returns:
    - tf.Variable, output tensor;
    """
    with tf.variable_scope(name):
        output_tensor = tf.layers.conv3d(input_tensor,
                                         filters=filters,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         use_bias=use_bias,
                                         name='conv3d',
                                         padding=padding,
                                         kernel_initializer=kernel_init_fn)

        output_tensor = activation(output_tensor)
    return output_tensor


def bn_conv3d(self, input_tensor, filters, kernel_size, name,
              strides=(1, 1, 1), padding='same', activation=tf.nn.relu,
              use_bias=False, kernel_init='xavier', is_training=True):
    """ Apply 3D convolution operation with batch normalization to input tensor.

    Args:
    - input_tensor: tf.Variable, input tensor;
    - filters: int, number of filters in the ouput tensor;
    - kernel_size: tuple(int, int, int), size of kernel
      of 3D convolution operation along 3 dimensions;
    - name: str, name of the layer that will be used as an argument of tf.variable_scope;
    - strides: tuple(int, int, int), size of strides along 3 dimensions;
    - padding: str, padding mode, can be 'same' or 'valid';
    - activation: tensorflow activation function that will be applied to
    output tensor;
    - use_bias: bool, whether use bias or not;

    Returns:
    - tf.Variable, output tensor;
    """
    with tf.variable_scope(name):
        output_tensor = tf.layers.conv3d(input_tensor,
                                         filters=filters,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         use_bias=use_bias,
                                         name='conv3d',
                                         padding=padding)

        output_tensor = tf.layers.batch_normalization(output_tensor, axis=-1,
                                                      training=is_training)
        output_tensor = activation(output_tensor)
    return output_tensor


def bn_dilated_conv3d(self, input_tensor, filters,
                      kernel_size, name, activation=tf.nn.relu,
                      dilation=(1, 1, 1), padding='same'):
    """ Apply 3D convolution operation with batch normalization to input tensor.

    Args:
    - input_tensor: tf.Variable, input tensor;
    - filters: int, number of filters in the ouput tensor;
    - kernel_size: tuple(int, int, int), size of kernel
      of 3D convolution operation along 3 dimensions;
    - name: str, name of the layer that will be used as an argument of tf.variable_scope;
    - dilation: tuple(int, int, int), size of dilation along 3 dimensions;
    - padding: str, padding mode, can be 'same' or 'valid';
    - activation: tensorflow activation function that will be applied to
    output tensor;

    Returns:
    - tf.Variable, output tensor;
    """

    in_filters = input_tensor.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        w = tf.get_variable(name='W', shape=(*kernel_size, in_filters, filters),
                            initializer=tf.contrib.layers.xavier_initializer())

        output_tensor = tf.nn.convolution(input_tensor, w,
                                          padding=padding.upper(),
                                          strides=(1, 1, 1),
                                          dilation_rate=dilation)

        output_tensor = tf.layers.batch_normalization(output_tensor,
                                                      axis=4,
                                                      training=is_training)

        output_tensor = activation(output_tensor)

    return output_tensor

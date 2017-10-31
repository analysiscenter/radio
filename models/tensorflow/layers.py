# pylint: disable=not-context-manager
# pylint: disable=too-many-arguments
"""Helper functions for creating layers """

import tensorflow as tf


def get_initializer(initializer):
    """ Get tensorflow initializer.

    Returns normal initializer or xavier initializer.

    Parameters
    ----------
    kernel_init : str
        initializer tyep; can be 'xavier' or 'normal'.

    Returns
    -------
    tensorflow initializer.
    """
    if initializer == 'xavier':
        kernel_init_fn = tf.contrib.layers.xavier_initializer
    elif initializer == 'normal':
        kernel_init_fn = tf.random_normal_initializer
    else:
        raise ValueError("Argument kernel_init must have 'str' type " +
                         "and be 'xavier' or 'normal'")


def selu(x):
    """ Selu activation function.

    Apply selu activation function to input tensor.

    Parameters
    ----------
    x : tf.Tensor
        input tensor.

    Returns
    -------
    tf.Tensor
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def conv3d(input_tensor, filters, kernel_size, name,
           strides=(1, 1, 1), padding='same', activation=tf.nn.relu,
           use_bias=True, kernel_init='xavier', is_training=True):
    """ Apply 3D convolution operation to input tensor.

    Parameters
    ----------
    input_tensor : tf.Tensor
        input tensor for 3D-convolution.
    filters : int
        number of filters in the ouput tensor.
    kernel_size : tuple(int, int, int)
        size of kernel for 3D-convolution operation along 3 dimensions.
    name : str
        name of the layer that will be used as an argument of tf.variable_scope.
    strides : tuple(int, int, int)
        size of strides along 3 dimensions required by tf.layers.conv3d.
    padding : str
        padding mode, can be 'same' or 'valid'.
    activation : tensorflow activation function
        this function will be applied to output tensor.
    use_bias : bool
        whether use bias or not.
    kernel_init : str
        initializer for kernel, can be 'xavier' or 'normal'.

    Returns
    -------
    tf.Tensor
        output tensor.
    """
    init_fn = get_initializer(kernel_init)
    with tf.variable_scope(name):
        output_tensor = tf.layers.conv3d(input_tensor,
                                         filters=filters,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         use_bias=use_bias,
                                         name='conv3d',
                                         padding=padding,
                                         kernel_initializer=init_fn)

        output_tensor = activation(output_tensor)
    return output_tensor


def bn_conv3d(input_tensor, filters, kernel_size, name,
              strides=(1, 1, 1), padding='same', activation=tf.nn.relu,
              use_bias=False, kernel_init='xavier', is_training=True):
    """ Apply 3D convolution operation with batch normalization to input tensor.

    Parameters
    ----------
    input_tensor : tf.Tensor
        input tensor for 3D-convolution.
    filters : int
        number of filters in the ouput tensor.
    kernel_size : tuple(int, int, int)
        size of kernel for 3D-convolution operation along 3 dimensions.
    name : str
        name of the layer that will be used as an argument of tf.variable_scope.
    strides : tuple(int, int, int)
        size of strides along 3 dimensions required by tf.layers.conv3d.
    padding : str
        padding mode, can be 'same' or 'valid'.
    activation : tensorflow activation function
        this function will be applied to output tensor.
    use_bias : bool
        whether use bias or not.
    kernel_init : str
        initializer for kernel, can be 'xavier' or 'normal'.
    is_training : tf.Tensor or bool
        whether model is in training state of prediction state.

    Returns
    -------
    tf.Tensor
        output tensor.
    """
    init_fn = get_initializer(kernel_init)
    with tf.variable_scope(name):
        output_tensor = tf.layers.conv3d(input_tensor,
                                         filters=filters,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         use_bias=use_bias,
                                         name='conv3d',
                                         padding=padding,
                                         kernel_initializer=init_fn)

        output_tensor = tf.layers.batch_normalization(output_tensor, axis=-1,
                                                      training=is_training)
        output_tensor = activation(output_tensor)
    return output_tensor


def bn_dilated_conv3d(input_tensor, filters,
                      kernel_size, name, activation=tf.nn.relu,
                      dilation=(1, 1, 1), padding='same',
                      is_training=True, kernel_init='xavier'):
    """ Apply 3D-dilated-convolution operation with batch normalization to input tensor.

    Parameters
    ----------
    input_tensor : tf.Tensor
        input tensor for 3D-convolution.
    filters : int
        number of filters in the ouput tensor.
    kernel_size : tuple(int, int, int)
        size of kernel for 3D-convolution operation along 3 dimensions.
    name : str
        name of the layer that will be used as an argument of tf.variable_scope.
    dilation : tuple(int, int, int)
        dilation rate along 3 dimensions.
    strides : tuple(int, int, int)
        size of strides along 3 dimensions required by tf.layers.conv3d.
    padding : str
        padding mode, can be 'same' or 'valid'.
    activation : tensorflow activation function
        this function will be applied to output tensor.
    use_bias : bool
        whether use bias or not.
    kernel_init : str
        initializer for kernel, can be 'xavier' or 'normal'.
    is_training : tf.Tensor or bool
        whether model is in training state of prediction state.

    Returns
    -------
    tf.Tensor
        output tensor.
    """

    in_filters = input_tensor.get_shape().as_list()[-1]
    init_fn = get_initializer(kernel_init)
    with tf.variable_scope(name):
        w = tf.get_variable(name='W', shape=(*kernel_size, in_filters, filters),
                            initializer=init_fn)

        output_tensor = tf.nn.convolution(input_tensor, w,
                                          padding=padding.upper(),
                                          strides=(1, 1, 1),
                                          dilation_rate=dilation)

        output_tensor = tf.layers.batch_normalization(output_tensor,
                                                      axis=4,
                                                      training=is_training)

        output_tensor = activation(output_tensor)

    return output_tensor


def global_average_pool3d(input_tensor, name):
    """ Apply global average pooling 3D operation to input tensor.

    Parameters
    ----------
    input_tensor : tf.Tensor
        input tensor.
    name : str
        name of layer that will be used as an argument of tf.variable_scope.

    Returns
    -------
    tf.Tensor
        output tensor.
    """
    with tf.variable_scope(name):
        output_layer = tf.reduce_mean(input_tensor, axis=(1, 2, 3))
    return output_layer

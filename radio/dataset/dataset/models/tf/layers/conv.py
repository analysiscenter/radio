""" Contains convolutional layers """
import tensorflow as tf


_CONV_LAYERS = {
    1: tf.layers.conv1d,
    2: tf.layers.conv2d,
    3: tf.layers.conv3d
}

def conv(inputs, *args, **kwargs):
    """ Nd convolution layer. Just a wrapper around ``tf.layers.conv1d``, ``conv2d``, ``conv3d``.

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor

    See also
    --------
    `tf.layers.conv1d <https://www.tensorflow.org/api_docs/python/tf/layers/conv1d>`_,
    `tf.layers.conv2d <https://www.tensorflow.org/api_docs/python/tf/layers/conv2d>`_,
    `tf.layers.conv3d <https://www.tensorflow.org/api_docs/python/tf/layers/conv3d>`_
    """
    dim = inputs.shape.ndims - 2
    layer_fn = _CONV_LAYERS[dim]
    return layer_fn(inputs, *args, **kwargs)

def conv1d_transpose(inputs, filters, kernel_size, strides=1, padding='valid', data_format='channels_last',
                     **kwargs):
    """ Transposed 1D convolution layer

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor
    filters : int
        number of filters in the ouput tensor
    kernel_size : int
        kernel size
    strides : int
        convolution stride. Default is 1.

    Returns
    -------
    tf.Tensor

    See also
    --------
    `tf.layers.conv2d_transpose <https://www.tensorflow.org/api_docs/python/tf/layers/conv2d_transpose>`_,
    `tf.layers.conv3d_transpose <https://www.tensorflow.org/api_docs/python/tf/layers/conv3d_transpose>`_
    """
    axis = 1 if data_format == 'channels_last' else 2
    up_tensor = tf.expand_dims(inputs, axis=axis)
    conv_output = tf.layers.conv2d_transpose(up_tensor, filters=filters, kernel_size=(1, kernel_size),
                                             strides=(1, strides), padding=padding, **kwargs)
    output = tf.squeeze(conv_output, [axis])
    return output

def conv_transpose(inputs, filters, kernel_size, strides, *args, **kwargs):
    """ Transposed Nd convolution layer

    Parameters
    ----------
    dim : int {1, 2, 3}
        number of dimensions
    inputs : tf.Tensor
        input tensor
    filters : int
        number of filters in the ouput tensor
    kernel_size : int
        kernel size
    strides : int
        convolution stride. Default is 1.

    Returns
    -------
    tf.Tensor

    See also
    --------
    :func:`.conv1d_transpose`,
    `tf.layers.conv2d_transpose <https://www.tensorflow.org/api_docs/python/tf/layers/conv2d_transpose>`_,
    `tf.layers.conv3d_transpose <https://www.tensorflow.org/api_docs/python/tf/layers/conv3d_transpose>`_
    """
    dim = inputs.shape.ndims - 2
    if dim == 1:
        output = conv1d_transpose(inputs, filters, kernel_size, strides, *args, **kwargs)
    elif dim == 2:
        output = tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides, *args, **kwargs)
    elif dim == 3:
        output = tf.layers.conv3d_transpose(inputs, filters, kernel_size, strides, *args, use_bias=False, **kwargs)
    return output


def separable_conv(inputs, filters, kernel_size, strides=1, padding='same', data_format='channels_last',
                   dilation_rate=1, depth_multiplier=1, activation=None, name=None, *args, **kwargs):
    """ Make Nd depthwise convolutions that acts separately on channels,
    followed by a pointwise convolution that mixes channels.

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor
    filters : int
        number of filters in the ouput tensor
    kernel_size : int
        kernel size
    strides : int
        convolution stride. Default is 1.
    padding : str
        padding mode, can be 'same' or 'valid'. Default - 'same',
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_last'.
    dilation_rate : int
        Default is 1.
    depth_multiplier : int
        The number of depthwise convolution output channels for each input channel.
        The total number of depthwise convolution output channels will be equal to
        ``num_filters_in`` * ``depth_multiplier``.
    activation : callable
        Default is `tf.nn.relu`.
    name : str
        The name of the layer

    Returns
    -------
    tf.Tensor

    """
    dim = inputs.shape.ndims - 2
    conv_layer = _CONV_LAYERS[dim]

    if dim == 2:
        return tf.layers.separable_conv2d(inputs, filters, kernel_size, strides, padding, data_format,
                                          dilation_rate, depth_multiplier, activation, name, *args, **kwargs)
    else:
        context = None
        if name is not None:
            context = tf.variable_scope(name)
            context.__enter__()

        inputs_shape = inputs.get_shape().as_list()
        axis = -1 if data_format == 'channels_last' else 1
        size = [-1] * (dim + 2)
        size[axis] = 1
        channels_in = inputs_shape[axis]

        depthwise_layers = []
        for channel in range(channels_in):
            start = [0] * (dim + 2)
            start[axis] = channel

            input_slice = tf.slice(inputs, start, size)
            slice_conv = conv_layer(input_slice, depth_multiplier, kernel_size, strides, padding, data_format,
                                    dilation_rate, activation, name='slice-%d' % channel, *args, **kwargs)
            depthwise_layers.append(slice_conv)

        # Concatenate the per-channel convolutions along the channel dimension.
        depthwise_conv = tf.concat(depthwise_layers, axis=axis)

        if channels_in * depth_multiplier != filters:
            output = conv_layer(depthwise_conv, filters, 1, 1, padding, data_format, 1, activation,
                                name='pointwise', *args, **kwargs)
        else:
            output = depthwise_conv

        if context is not None:
            context.__exit__(None, None, None)

    return output


def separable_conv1d(inputs, filters, kernel_size, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=1, depth_multiplier=1, activation=None, name=None, *args, **kwargs):
    """ Make 1d depthwise convolutions that acts separately on channels,
    followed by a pointwise convolution that mixes channels.

    See also
    --------
    :func:`.separable_conv`
    """
    return separable_conv(inputs, filters, kernel_size, strides, padding, data_format,
                          dilation_rate, depth_multiplier, activation, name, *args, **kwargs)

def separable_conv2d(inputs, filters, kernel_size, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=1, depth_multiplier=1, activation=None, name=None, *args, **kwargs):
    """ Make 2d depthwise convolutions that acts separately on channels,
    followed by a pointwise convolution that mixes channels.

    See also
    --------
    :func:`.separable_conv`,
    `tf.layers.separable_conv2d <https://www.tensorflow.org/api_docs/python/tf/layers/separable_conv2d>`_

    Notes
    -----
    `tf.layers.separable_conv2d <https://www.tensorflow.org/api_docs/python/tf/layers/separable_conv2d>`_
    is implemented using CUDA and, as a result, much faster.
    """
    return separable_conv(inputs, filters, kernel_size, strides, padding, data_format,
                          dilation_rate, depth_multiplier, activation, name, *args, **kwargs)

def separable_conv3d(inputs, filters, kernel_size, strides=1, padding='same', data_format='channels_last',
                     dilation_rate=1, depth_multiplier=1, activation=None, name=None, *args, **kwargs):
    """ Make 3d depthwise convolutions that acts separately on channels,
    followed by a pointwise convolution that mixes channels.

    See also
    --------
    :func:`.separable_conv`
    """
    return separable_conv(inputs, filters, kernel_size, strides, padding, data_format,
                          dilation_rate, depth_multiplier, activation, name, *args, **kwargs)

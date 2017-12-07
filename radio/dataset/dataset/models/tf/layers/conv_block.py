""" Contains convolution layers """
# pylint:disable=too-many-statements
import logging
import tensorflow as tf

from .core import mip, flatten
from .conv import conv_transpose, separable_conv
from .pooling import max_pooling, average_pooling, global_max_pooling, global_average_pooling, fractional_max_pooling


ND_LAYERS = {
    'activation': None,
    'dense': tf.layers.dense,
    'conv': [tf.layers.conv1d, tf.layers.conv2d, tf.layers.conv3d],
    'transposed_conv': conv_transpose,
    'separable_conv':separable_conv,
    'max_pooling': max_pooling,
    'average_pooling': average_pooling,
    'fractional_max_pooling': fractional_max_pooling,
    'global_max_pooling': global_max_pooling,
    'global_average_pooling': global_average_pooling,
    'batch_norm': tf.layers.batch_normalization,
    'dropout': tf.layers.dropout,
    'mip': mip
}

C_LAYERS = {
    'a': 'activation',
    'f': 'dense',
    'c': 'conv',
    't': 'transposed_conv',
    's': 'separable_conv',
    'p': 'max_pooling',
    'v': 'average_pooling',
    'r': 'fractional_max_pooling',
    'P': 'global_max_pooling',
    'V': 'global_average_pooling',
    'n': 'batch_norm',
    'd': 'dropout',
    'm': 'mip'
}

_LAYERS_KEYS = str(list(C_LAYERS.keys()))
_GROUP_KEYS = (
    _LAYERS_KEYS
    .replace('t', 'c')
    .replace('s', 'c')
    .replace('v', 'p')
)
C_GROUPS = dict(zip(_LAYERS_KEYS, _GROUP_KEYS))

def _get_layer_fn(fn, dim):
    f = ND_LAYERS[fn]
    return f if callable(f) or f is None else f[dim-1]

def _unpack_args(args, layer_no, layers_max):
    new_args = {}
    for arg in args:
        if isinstance(args[arg], list) and layers_max > 1:
            arg_value = args[arg][layer_no]
        else:
            arg_value = args[arg]
        new_args.update({arg: arg_value})

    return new_args

def conv_block(inputs, layout='', filters=0, kernel_size=3, name=None,
               strides=1, padding='same', data_format='channels_last', dilation_rate=1, depth_multiplier=1,
               activation=tf.nn.relu, pool_size=2, pool_strides=2, dropout_rate=0., is_training=True, **kwargs):
    """ Complex multi-dimensional block with a sequence of convolutions, batch normalization, activation, pooling,
    dropout and even dense layers.

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor
    layout : str
        a sequence of layers:

        - c - convolution
        - t - transposed convolution
        - s - separable convolution
        - f - dense (fully connected)
        - n - batch normalization
        - a - activation
        - p - max pooling
        - v - average pooling
        - r - fractional max pooling
        - P - global max pooling
        - V - global average pooling
        - d - dropout
        - m - maximum intensity projection (:func:`.layers.mip`)

        Default is ''.
    filters : int
        the number of filters in the ouput tensor
    kernel_size : int
        kernel size
    name : str
        name of the layer that will be used as a scope.
    units : int
        the number of units in the dense layer
    strides : int
        Default is 1.
    padding : str
        padding mode, can be 'same' or 'valid'. Default - 'same',
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_last'.
    dilation_rate: int
        Default is 1.
    activation : callable
        Default is `tf.nn.relu`.
    pool_size : int
        Default is 2.
    pool_strides : int
        Default is 2.
    dropout_rate : float
        Default is 0.
    is_training : bool or tf.Tensor
        Default is True.

    dense : dict
        parameters for dense layers, like initializers, regularalizers, etc
    conv : dict
        parameters for convolution layers, like initializers, regularalizers, etc
    transposed_conv : dict
        parameters for transposed conv layers, like initializers, regularalizers, etc
    batch_norm : dict or None
        parameters for batch normalization layers, like momentum, intiializers, etc
        If None or inculdes parameters 'off' or 'disable' set to True or 1,
        the layer will be excluded whatsoever.
    max_pooling : dict
        parameters for max_pooling layers, like initializers, regularalizers, etc
    dropout : dict or None
        parameters for dropout layers, like noise_shape, etc
        If None or inculdes parameters 'off' or 'disable' set to True or 1,
        the layer will be excluded whatsoever.

    Returns
    -------
    output tensor : tf.Tensor

    Notes
    -----
    When ``layout`` includes several layers of the same type, each one can have its own parameters,
    if corresponding args are passed as lists (not tuples).

    Examples
    --------
    A simple block: 3x3 conv, batch norm, relu, 2x2 max-pooling with stride 2::

        x = conv_block(x, 32, 3, layout='cnap')

    A canonical bottleneck block (1x1, 3x3, 1x1 conv with relu in-between)::

        x = conv_block(x, [64, 64, 256], [1, 3, 1], layout='cacac')

    A complex Nd block:

    - 5x5 conv with 32 filters
    - relu
    - 3x3 conv with 32 filters
    - relu
    - 3x3 conv with 64 filters and a spatial stride 2
    - relu
    - batch norm
    - dropout with rate 0.15

    ::

        x = conv_block(x, [32, 32, 64], [5, 3, 3], layout='cacacand', strides=[1, 1, 2], dropout_rate=.15)
    """
    layout = layout or ''
    layout = layout.replace(' ', '')
    if len(layout) == 0:
        logging.warning('conv_block: layout is empty, so there is nothing to do, just returning inputs.')
        return inputs

    dim = inputs.shape.ndims - 2
    if not isinstance(dim, int) or dim > 3:
        raise ValueError("Number of dimensions of the inputs tensor should be 1, 2 or 3, but given %d" % dim)

    context = None
    if name is not None:
        context = tf.variable_scope(name, reuse=kwargs.get('reuse'))
        context.__enter__()

    layout_dict = {}
    for layer in layout:
        if C_GROUPS[layer] not in layout_dict:
            layout_dict[C_GROUPS[layer]] = [-1, 0]
        layout_dict[C_GROUPS[layer]][1] += 1

    tensor = inputs
    for i, layer in enumerate(layout):

        layout_dict[C_GROUPS[layer]][0] += 1
        layer_name = C_LAYERS[layer]
        layer_fn = _get_layer_fn(layer_name, dim)

        if layer == 'a':
            args = dict(activation=activation)
            layer_fn = _unpack_args(args, *layout_dict[C_GROUPS[layer]])['activation']
            if layer_fn is not None:
                tensor = layer_fn(tensor)
        else:
            layer_args = kwargs.get(layer_name, {})
            skip_layer = layer_args is None or layer_args is False or \
                         isinstance(layer_args, dict) and 'disable' in layer_args and layer_args.pop('disable')

            if skip_layer:
                pass
            elif layer == 'f':
                if tensor.shape.ndims > 2:
                    tensor = flatten(tensor)
                units = kwargs.get('units')
                if units is None:
                    raise ValueError('units cannot be None if layout includes dense layers')
                args = dict(units=units)

            elif layer == 'c':
                args = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            data_format=data_format, dilation_rate=dilation_rate)
                if filters is None or filters == 0:
                    raise ValueError('filters cannot be None or 0 if layout includes convolutional layers')

            elif layer == 's':
                args = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            data_format=data_format, dilation_rate=dilation_rate, depth_multiplier=depth_multiplier)
                if filters is None or filters == 0:
                    raise ValueError('filters cannot be None or 0 if layout includes convolutional layers')

            elif layer == 't':
                args = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            data_format=data_format)
                if filters is None or filters == 0:
                    raise ValueError('filters cannot be None or 0 if layout includes convolutional layers')

            elif layer == 'n':
                axis = -1 if data_format == 'channels_last' else 1
                args = dict(fused=True, axis=axis, training=is_training)

            elif layer == 'r':
                args = dict(pooling_ratio=kwargs.get('pooling_ratio', 1.4142),
                            pseudo_random=kwargs.get('pseudo_random', False),
                            overlapping=kwargs.get('overlapping', False),
                            data_format=data_format)

            elif C_GROUPS[layer] == 'p':
                args = dict(pool_size=pool_size, strides=pool_strides, padding=padding,
                            data_format=data_format)

            elif layer == 'd':
                if dropout_rate:
                    args = dict(rate=dropout_rate, training=is_training)
                else:
                    skip_layer = True

            elif layer in ['P', 'V']:
                args = dict(data_format=data_format)

            elif layer == 'm':
                args = dict(data_format=data_format)

            if not skip_layer:
                args = {**args, **layer_args}
                args = _unpack_args(args, *layout_dict[C_GROUPS[layer]])

                with tf.variable_scope('layer-%d' % i):
                    tensor = layer_fn(inputs=tensor, **args)

    if context is not None:
        context.__exit__(None, None, None)

    return tensor

""" Contains different useful tensorflow functions. """

import numpy as np
import tensorflow as tf


def repeat_tensor(input_tensor, times):
    """ Repeat tensor given times along axes.

    Args:
    - input_tensor: tf.Tensor, input tensor;
    - times: tuple(int, int,..., int) number of times to repeat
    tensor along each axis;

    Return:
    - tf.Tensor, repeated tensor;
    """
    source_shape = get_shape(input_tensor)
    x = tf.expand_dims(input_tensor, axis=len(source_shape))
    x = tf.tile(x, [1, *times])

    new_shape = tuple(np.array(source_shape[1:]) * np.array(times[1:]))
    x = tf.reshape(x, shape=(-1, *new_shape))
    return x


def get_shape(input_tensor):
    """ Return full shape of the input tensor represented by tuple of ints.

    Args:
    - input_tensor: tf.Variable, input_tensor;

    Return:
    - shape of input_tensor, tuple(int);
    """
    return input_tensor.get_shape().as_list()


def batch_size(input_tensor):
    """ Return batch size of the input tensor represented by first dimension.

    Args:
    - input_tensor: tf.Variable, input_tensor.

    Return:
    - number of items in the batch, int;
    """
    return get_shape(input_tensor)[0]


def num_channels(input_tensor):
    """ Get number of channels in input tensor.

    Args:
    - input_tensor, tf.Tensor;

    Returns:
    - number of channels;

    NOTE: channels last ordering is used;
    """
    return get_shape(input_tensor)[-1]


def split_channels(input_tensor, size):
    """ Split channels of input tensor into groups of given size.

    Args:
    - input_tensor: tf.Tensor, input tensor;
    - size: int, size of each group;

    Returns:
    - list of tensors, each corresponds to channels group.
    """
    in_filters = num_channels(input_tensor)
    if in_filters <= size:
        return input_tensor
    a, b = int(in_filters / size), int(in_filters % size)
    main = list(tf.split(input_tensor[..., : a * size], a, axis=len(input_tensor.get_shape()) - 1))
    if b != 0:
        main.append(input_tensor[..., a * size: ])
    return main


def channels_rnd_shuffle(input_tensor):
    """ Perform random shuffle of channels in input tensor.

    Args:
    - input_tensor: tf.Tensor, input tensor;

    Returns:
    - tf.Tensor, tensor with random shuffle of channels.
    """
    num_filters = num_channels(input_tensor)
    indices = np.random.permutation(num_filters)
    tensors_list = []
    for i in indices:
        tensors_list.append(input_tensor[..., i, tf.newaxis])
    return tf.concat(tensors_list, axis=-1)

""" Contains custom losses """
import tensorflow as tf

from ..layers import flatten


def dice(targets, predictions):
    """ Dice coefficient

    Parameters
    ----------
    targets : tf.Tensor
        tensor with target values

    predictions : tf.Tensor
        tensor with predicted values

    Returns
    -------
    average loss : tf.Tensor with a single element
    """
    e = 1e-6
    intersection = flatten(targets * predictions)
    loss = -tf.reduce_mean((2. * intersection + e) / (flatten(targets) + flatten(predictions) + e))
    return loss

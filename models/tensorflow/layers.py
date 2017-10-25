"""Helper functions for creating layers """

import tensorflow as tf

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

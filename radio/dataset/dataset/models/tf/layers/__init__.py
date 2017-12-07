""" Custom tf layers and operations """
import numpy as np
import tensorflow as tf

from .core import flatten, flatten2d, maxout, mip
from .conv_block import conv_block
from .conv import conv1d_transpose, conv_transpose, separable_conv, separable_conv1d, separable_conv2d, separable_conv3d
from .pooling import max_pooling, average_pooling, global_average_pooling, global_max_pooling, fractional_max_pooling

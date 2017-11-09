# pylint: disable=anomalous-backslash-in-string
# pylint: disable=not-context-manager
""" Contains implementation of VNet architecture with dilated convolutions in tensorflow. """

import math
import numpy as np
import tensorflow as tf
from ..tf_model import TFModelCT
from ..layers import bn_conv3d, bn_dilated_conv3d
from ..utils import repeat_tensor


class TFDilatedVnet(TFModelCT):

    """ This class implements VNet architecture with dilated convolutions using tensorflow.

    Attributes
    ----------
    config : dict
        config dictionary from dataset pipeline
        see configuring model section of dataset module
        https://analysiscenter.github.io/dataset/intro/models.html.
    name : str
        name of the model, can be specified in config dict.

    NOTE
    ----
    This class is descendant of TFModel class from dataset.models.*.
    """

    def upsampling3d(self, input_tensor, times, name):
        """ Apply 3D upsampling operation to input tensor.

        This operation is a kind of reverse operation for maxpooling3D.

        Parameters
        ----------
        input_tensor : tf.Tensor
            input tensor.
        times : tuple(int, int, int)
            number of times to repeat values along each spatial axis.

        Returns
        -------
        tf.Tensor
            output tensor.

        Note
        ----
        This layer does not perform repeat operation
        along channels and batch axes.
        """
        if isinstance(times, (list, tuple, np.ndarray)):
            _times = tuple(times)
        else:
            _times = (times, times, times)

        if len(_times) != 3:
            raise ValueError("Times must be tuple, list or ndarray of size 3")

        with tf.variable_scope(name):
            return repeat_tensor(input_tensor, (1, *_times, 1))

    def bottleneck_block(self, input_tensor, filters, scope, dilation=(1, 1, 1), padding='same'):
        """ Apply bottleneck block transform to input tensor.

        Parameters
        ----------
        input_tensor : tf.Tensor
            input tensor.
        filters : int
            number of filters.
        scope : str
            name scope of the layer.
        dilation : tuple(int, int, int)
            dilation rate along spatial axes.
        padding : str
            padding mode, can be 'valid' or 'same'.

        Returns
        -------
        tf.Tensor
            output tensor.
        """
        with tf.variable_scope(scope):
            n, m = math.ceil(filters / 2), math.floor(filters / 2)
            conv1 = bn_conv3d(input_tensor, filters, (1, 1, 1),
                              padding=padding, name='Conv3D_1x1x1',
                              is_training=self.is_training)

            conv2 = bn_conv3d(conv1, n, (3, 3, 3),
                              padding=padding, name='Conv3D_3x3x3',
                              is_training=self.is_training)

            conv2_dilated = bn_dilated_conv3d(conv1, m, (3, 3, 3),
                                              dilation=dilation,
                                              padding=padding,
                                              name='Conv3D_dilated',
                                              is_training=self.is_training)

            conv_concated = tf.concat([conv2, conv2_dilated], axis=4)

        return conv_concated

    def reduction_block(self, input_tensor, filters, scope, dilation=(1, 1, 1),
                        pool_size=(2, 2, 2), padding='same'):
        """ Apply reduction block transform to input tensor.

        This layer consists of two 3D-convolutional layers with batch normalization
        before 'relu' activation and max_pooling3d layer in the end.
        Middle layer contains two separate convolutions:
        one with dilation_rate=(2, 2, 2) and another with dilation_rate=(1, 1, 1).

        Parameters
        ----------
        input_tensor : tf.Tensor
            input tensor.
        filters : int
            number of filters in first and second covnolutions.
        scope : str
            name of scope for this reduction block.
        dilation : tuple(int, int, int)
            dilation rate along spatial axes.
        pool_size : tuple(int, int, int)
            size of pooling kernel along three axis.
        padding : str
            padding mode for convolutions, can be 'same' or 'valid'.

        Returns
        -------
        tuple(tf.Tensor, tf.Tensor)
            first tensor corresponds to the output of concatenation
            operation before max_pooling, second -- after max_pooling.
        """
        with tf.variable_scope(scope):
            n, m = math.ceil(filters / 2), math.floor(filters / 2)
            conv1 = bn_conv3d(input_tensor, filters, (1, 1, 1),
                              padding=padding, name='Conv3D_1x1x1',
                              is_training=self.is_training)

            conv2 = bn_conv3d(conv1, n, (3, 3, 3),
                              padding=padding, name='Conv3D_3x3x3',
                              is_training=self.is_training)

            conv2_dilated = bn_dilated_conv3d(conv1, m, (3, 3, 3),
                                              dilation=dilation, padding=padding,
                                              name='Conv3D_dilated',
                                              is_training=self.is_training)

            conv_concated = tf.concat([conv2, conv2_dilated], axis=4)
            max_pool = tf.layers.max_pooling3d(conv_concated, pool_size=(2, 2, 2),
                                               strides=(2, 2, 2), padding='same',
                                               name='max_pool3d')
        return conv_concated, max_pool

    def upsampling_block(self, input_tensor, skip_connect_tensor, filters,
                         scope, dilation=(2, 2, 2), padding='same'):
        """ Apply upsampling transform to two input tensors.

        First of all, UpSampling3D transform is applied to input_tensor. Then output
        tensor of this operation is concatenated with skip_connect_tensor. After this
        two 3D-convolutions with batch normalization before 'relu' activation
        are applied.

        Parameters
        ----------
        input_tensor : tf.Tensor
            input tensor from previous layer.
        skip_connect_tensor : tf.Tensor
            input tensor from simmiliar layer from reduction branch of VNet.
        filters : int
            number of filters in convolutional layers.
        scope : str
            name of scope for this block.
        dilation : tuple(int, int, int)
            dilation rate along spatial axes.
        padding : str
            padding mode for convolutions, can be 'same' or 'valid'.

        Returns
        -------
        tf.Tensor
            output tensor.
        """
        with tf.variable_scope(scope):
            n, m = math.ceil(filters / 2), math.floor(filters / 2)
            upsample_tensor = self.upsampling3d(input_tensor,
                                                times=(2, 2, 2),
                                                name='upsample3d')

            upsample_tensor = tf.concat([upsample_tensor, skip_connect_tensor], axis=4)

            conv1 = bn_conv3d(upsample_tensor, n, (3, 3, 3),
                              padding=padding, name='Conv3D_1x1x1',
                              is_training=self.is_training)

            conv1_dilated = bn_dilated_conv3d(conv1, m, (3, 3, 3),
                                              dilation=dilation,
                                              padding=padding,
                                              name='Conv3D_dilated_I',
                                              is_training=self.is_training)

            conv1_stacked = tf.concat([conv1, conv1_dilated], axis=4)

            conv2 = bn_conv3d(conv1_stacked, n, (3, 3, 3),
                              padding=padding, name='Conv3D_3x3x3',
                              is_training=self.is_training)

            conv2_dilated = bn_dilated_conv3d(conv1_stacked, m, (3, 3, 3),
                                              dilation=dilation, padding=padding,
                                              name='Conv3D_dilated_II',
                                              is_training=self.is_training)

            conv2_stacked = tf.concat([conv2, conv2_dilated], axis=4)
        return conv2_stacked

    def _build(self, *args, **kwargs):
        """ Build vnet with dilated convolutions model implemented in tensorflow. """
        input_tensor = tf.placeholder(shape=(None, 32, 64, 64, 1), dtype=tf.float32)
        y_true = tf.placeholder(shape=(None, 32, 64, 64, 1), dtype=tf.float32, name='targets')

        # Downsampling or reduction layers: ReductionBlock_A, ReductionBlock_B, ReductionBlock_C, ReductionBlock_D
        # block_A has shape (None, 32, 64, 64, 32), reduct_block_A has shape (None, 16, 32, 32, 32)
        block_A, reduct_block_A = self.reduction_block(input_tensor, 32,
                                                       scope='DownBlock_A')

        # block_B has shape (None, 16, 32, 32, 64), reduct_block_B has shape (None, 8, 16, 16, 64)
        block_B, reduct_block_B = self.reduction_block(reduct_block_A, 64,
                                                       scope='DownBlock_B')

        # block_C has shape (None, 8, 16, 16, 128), reduct_block_C has shape (None, 4, 8, 8, 128)
        block_C, reduct_block_C = self.reduction_block(reduct_block_B, 128,
                                                       scope='DownBlock_C')

        # block_D has shape (None, 4, 8, 8, 256), reduct_block_D has shape (None, 2, 4, 4, 256)
        block_D, reduct_block_D = self.reduction_block(reduct_block_C, 256,
                                                       scope='DownBlock_D')

        # Bottleneck layer
        # bottleneck_block has shape (None, 2, 4, 4, 512)
        bottleneck_block = self.bottleneck_block(reduct_block_D, 512, 'BottleNeckBlock')

        # Upsampling Layers: UpsamplingBlock_D, UpsamplingBlock_C, UpsamplingBlock_B, UpsamplingBlock_A
        # upsample_block_C has shape (None, 4, 8, 8, 256)
        upsample_block_D = self.upsampling_block(bottleneck_block, block_D,
                                                 256, scope='UpBlock_D')

        # upsample_block_C has shape (None, 8, 16, 16, 128)
        upsample_block_C = self.upsampling_block(upsample_block_D, block_C,
                                                 128, scope='UpBlock_C')

        # upsample_block_B has shape (None, 16, 32, 32, 64)
        upsample_block_B = self.upsampling_block(upsample_block_C, block_B,
                                                 64, scope='UpBlock_B')

        # upsample_block_A has shape (None, 32, 64, 64, 32)
        upsample_block_A = self.upsampling_block(upsample_block_B, block_A,
                                                 32, scope='UpBlock_A')

        y_pred = bn_conv3d(upsample_block_A, 1, (1, 1, 1), name='final_conv',
                           activation=tf.nn.sigmoid, padding='same',
                           is_training=self.is_training)

        y_pred = tf.identity(y_pred, name='predictions')

        self.store_to_attr('y', y_true)
        self.store_to_attr('x', input_tensor)
        self.store_to_attr('y_pred', y_pred)
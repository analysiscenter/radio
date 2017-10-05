import math
import numpy as np
import tensorflow as tf
from .tf_model import TFModel, restore_nodes


def get_shape(input_tensor):
    """ Get shape of input tensor.

    Args:
    - input_tensor: tf.Tensor, input tensor;

    Returns:
    - tf.Tensor, output tensor;
    """
    return input_tensor.get_shape().as_list()


def repeat(input_tensor, times):
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


class TFDilatedUnet(TFModel):

    def upsampling3d(self, input_tensor, times, name):
        """ Apply 3D upsampling operation to input tensor.

        This operation is a kind of reverse operation for maxpooling3D.

        Args:
        - input_tensor: tf.Tensor, input tensor;
        - times: tuple(int, int, int), number of times to repeat values
        along each spatial axis;

        NOTE: this layer does not perform repeat operation
        along channels and batch axes.

        Returns:
        - tf.Tensor, output tensor;
        """
        if isinstance(times, (list, tuple, np.ndarray)):
            _times = tuple(times)
        else:
            _times = (times, times, times)

        if len(_times) != 3:
            raise ValueError("Times must be tuple, list or ndarray of size 3")

        with tf.variable_scope(name):
            return repeat(input_tensor, (1, *_times, 1))

    def bottleneck_block(self, input_tensor, filters, scope, dilation=(1, 1, 1),
                        pool_size=(2, 2, 2), padding='same'):

        with tf.variable_scope(scope):
            n, m = math.ceil(filters / 2), math.floor(filters / 2)
            conv1 = self.bn_conv3d(input_tensor, filters, (1, 1, 1),
                                   padding=padding, name='Conv3D_1x1x1')

            conv2 = self.bn_conv3d(conv1, n, (3, 3, 3),
                                   padding=padding, name='Conv3D_3x3x3')

            conv2_dilated = self.bn_dilated_conv3d(conv1, m, (3, 3, 3),
                                                   dilation=dilation, padding=padding,
                                                   name='Conv3D_dilated')

            conv_concated = tf.concat([conv2, conv2_dilated], axis=4)

        return conv_concated

    def reduction_block(self, input_tensor, filters, scope, dilation=(1, 1, 1),
                        pool_size=(2, 2, 2), padding='same'):
        """ Apply reduction block transform to input tensor.

        This layer consists of two 3D-convolutional layers with batch normalization
        before 'relu' activation and max_pooling3d layer in the end.

        Schematically this block can be represented like this:
        =======================================================================
        => Conv3D{3x3x3}[1:1:1](filters) => BatchNorm(filters_axis) => Relu =>
        => Conv3D{3x3x3}[1:1:1](filters) => BatchNorm(filters_axis) => Relu =>
        => MaxPooling3D{pool_size}[2:2:2]
        =======================================================================

        Args:
        - input_tensor: keras tensor, input tensor;
        - filters: int, number of filters in first and second covnolutions;
        - scope: str, name of scope for this reduction block;
        - dilation: tuple(int, int, int), dilation rate along spatial axes;
        - pool_size: tuple(int, int, int), size of pooling kernel along three axis;
        - padding: str, padding mode for convolutions, can be 'same' or 'valid';

        Returns:
        - ouput tensor;
        """
        with tf.variable_scope(scope):
            n, m = math.ceil(filters / 2), math.floor(filters / 2)
            conv1 = self.bn_conv3d(input_tensor, filters, (1, 1, 1),
                                   padding=padding, name='Conv3D_1x1x1')

            conv2 = self.bn_conv3d(conv1, n, (3, 3, 3),
                                   padding=padding, name='Conv3D_3x3x3')

            conv2_dilated = self.bn_dilated_conv3d(conv1, m, (3, 3, 3),
                                                   dilation=dilation, padding=padding,
                                                   name='Conv3D_dilated')

            conv_concated = tf.concat([conv2, conv2_dilated], axis=4)
            max_pool = tf.layers.max_pooling3d(conv_concated, pool_size=(2, 2, 2),
                                               strides=(2, 2, 2), padding='same',
                                               name='max_pool3d')
        return conv_concated, max_pool

    def upsampling_block(self, input_tensor, scip_connect_tensor, filters,
                         scope, dilation=(2, 2, 2), padding='same'):

        with tf.variable_scope(scope):
            n, m = math.ceil(filters / 2), math.floor(filters / 2)
            upsample_tensor = self.upsampling3d(input_tensor,
                                                times=(2, 2, 2),
                                                name='upsample3d')

            upsample_tensor = tf.concat([upsample_tensor, scip_connect_tensor], axis=4)

            conv1 = self.bn_conv3d(upsample_tensor, n, (3, 3, 3),
                                   padding=padding, name='Conv3D_1x1x1')

            conv1_dilated = self.bn_dilated_conv3d(conv1, m, (3, 3, 3),
                                                   dilation=dilation, padding=padding,
                                                   name='Conv3D_dilated_I')

            conv1_stacked = tf.concat([conv1, conv1_dilated], axis=4)

            conv2 = self.bn_conv3d(conv1_stacked, n, (3, 3, 3),
                                   padding=padding, name='Conv3D_3x3x3')

            conv2_dilated = self.bn_dilated_conv3d(conv1_stacked, m, (3, 3, 3),
                                                   dilation=dilation, padding=padding,
                                                   name='Conv3D_dilated_II')

            conv2_stacked = tf.concat([conv2, conv2_dilated], axis=4)
        return conv2_stacked

    @restore_nodes('input', 'y_true', 'y_pred')
    def build_model(self):

        input_tensor = tf.placeholder(shape=(None, 32, 64, 64, 1), dtype=tf.float32)
        y_true = tf.placeholder(shape=(None, 32, 64, 64, 1), dtype=tf.float32)

        # Downsampling or reduction layers: ReductionBlock_A, ReductionBlock_B, ReductionBlock_C, ReductionBlock_D
        # block_A has shape (None, 32, 64, 64, 32), reduct_block_A has shape (None, 16, 32, 32, 32)
        block_A, reduct_block_A = self.reduction_block(input_tensor, 32,
                                                       scope='ReductionBlock_A')

        # block_B has shape (None, 16, 32, 32, 64), reduct_block_B has shape (None, 8, 16, 16, 64)
        block_B, reduct_block_B = self.reduction_block(reduct_block_A, 64,
                                                       scope='ReductionBlock_B')

        # block_C has shape (None, 8, 16, 16, 128), reduct_block_C has shape (None, 4, 8, 8, 128)
        block_C, reduct_block_C = self.reduction_block(reduct_block_B, 128,
                                                       scope='ReductionBlock_C')

        # block_D has shape (None, 4, 8, 8, 256), reduct_block_D has shape (None, 2, 4, 4, 256)
        block_D, reduct_block_D = self.reduction_block(reduct_block_C, 256,
                                                       scope='ReductionBlock_D')

        # Bottleneck layer
        # bottleneck_block has shape (None, 2, 4, 4, 512)
        bottleneck_block = self.bottleneck_block(reduct_block_D, 512, 'BottleNeckBlock')

        # Upsampling Layers: UpsamplingBlock_D, UpsamplingBlock_C, UpsamplingBlock_B, UpsamplingBlock_A
        # upsample_block_C has shape (None, 4, 8, 8, 256)
        upsample_block_D = self.upsampling_block(bottleneck_block, block_D,
                                                 256, scope='UpsamplingBlock_D')

        # upsample_block_C has shape (None, 8, 16, 16, 128)
        upsample_block_C = self.upsampling_block(upsample_block_D, block_C,
                                                 128, scope='UpsamplingBlock_C')

        # upsample_block_B has shape (None, 16, 32, 32, 64)
        upsample_block_B = self.upsampling_block(upsample_block_C, block_B,
                                                 64, scope='UpsamplingBlock_B')

        # upsample_block_A has shape (None, 32, 64, 64, 32)
        upsample_block_A = self.upsampling_block(upsample_block_B, block_A,
                                                 32, scope='UpsamplingBlock_A')

        y_pred = self.bn_conv3d(upsample_block_A, 1, (1, 1, 1), name='final_conv',
                                activation=tf.nn.sigmoid, padding='same')

        return input_tensor, y_true, y_pred

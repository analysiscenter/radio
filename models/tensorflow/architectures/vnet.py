import math
import numpy as np
import tensorflow as tf
from ..tf_model import TFModel, restore_nodes
from ..utils import repeat_tensor


class TFVnet(TFModel):

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, **kwargs)

    def upsampling3d(self, input_tensor, times, name):
        if isinstance(times, (list, tuple, np.ndarray)):
            _times = tuple(times)
        else:
            _times = (times, times, times)

        if len(_times) != 3:
            raise ValueError("Times must be tuple, list or ndarray of size 3")

        with tf.variable_scope(name):
            return repeat_tensor(input_tensor, (1, *_times, 1))

    def bottleneck_block(self, input_tensor, filters, scope, padding='same'):

        with tf.variable_scope(scope):
            conv1 = self.conv3d_act_bn(
                input_tensor,
                filters, (3, 3, 3),
                padding=padding,
                name='Conv3D_3x3x3_I')

            conv2 = self.conv3d_act_bn(
                conv1,
                filters, (3, 3, 3),
                padding=padding,
                name='Conv3D_3x3x3_II')

        return conv2

    def reduction_block(self,
                        input_tensor,
                        filters,
                        scope,
                        pool_size=(2, 2, 2),
                        padding='same'):

        with tf.variable_scope(scope):
            conv1 = self.conv3d_act_bn(
                input_tensor,
                filters, (3, 3, 3),
                padding=padding,
                name='Conv3D_3x3x3_I')

            conv2 = self.conv3d_act_bn(
                conv1,
                filters, (3, 3, 3),
                padding=padding,
                name='Conv3D_3x3x3_II')

            mxpool = tf.layers.max_pooling3d(
                conv2,
                pool_size=(2, 2, 2),
                strides=(2, 2, 2),
                padding=padding,
                name='max_pool3d')

        return conv2, mxpool

    def upsampling_block(self,
                         input_tensor,
                         scip_connect_tensor,
                         filters,
                         scope,
                         padding='same'):

        with tf.variable_scope(scope):
            upsample_tensor = self.upsampling3d(
                input_tensor, times=(2, 2, 2), name='upsample3d')

            upsample_tensor = tf.concat(
                [upsample_tensor, scip_connect_tensor], axis=4)

            conv1 = self.conv3d_act_bn(
                upsample_tensor,
                filters, (3, 3, 3),
                padding=padding,
                name='Conv3D_1x1x1_I')

            conv2 = self.conv3d_act_bn(
                conv1,
                filters, (3, 3, 3),
                padding=padding,
                name='Conv3D_1x1x1_II')

        return conv1

    @restore_nodes('input', 'y_true', 'y_pred')
    def build_model(self):

        input_tensor = tf.placeholder(
            shape=(None, 32, 64, 64, 1), dtype=tf.float32)
        y_true = tf.placeholder(shape=(None, 32, 64, 64, 1), dtype=tf.float32)

        # Downsampling or reduction layers: ReductionBlock_A, ReductionBlock_B, ReductionBlock_C, ReductionBlock_D
        # (32, 64, 64) (16, 32, 32)
        block_A, reduct_block_A = self.reduction_block(
            input_tensor, 32, scope='ReductionBlock_A')

        # (16, 32, 32) (8, 16, 16)
        block_B, reduct_block_B = self.reduction_block(
            reduct_block_A, 64, scope='ReductionBlock_B')

        # (8, 16, 16) (4, 8, 8)
        block_C, reduct_block_C = self.reduction_block(
            reduct_block_B, 128, scope='ReductionBlock_C')

        # (4, 8, 8) (2, 4, 4)
        block_D, reduct_block_D = self.reduction_block(
            reduct_block_C, 256, scope='ReductionBlock_D')

        # Bottleneck layer
        # (2, 4, 4)
        bottleneck_block = self.bottleneck_block(reduct_block_D, 512,
                                                 'BottleNeckBlock')

        # Upsampling Layers: UpsamplingBlock_D, UpsamplingBlock_C, UpsamplingBlock_B, UpsamplingBlock_A
        # (4, 8, 8)
        upsample_block_D = self.upsampling_block(
            bottleneck_block,
            block_D,
            filters=256,
            scope='UpsamplingBlock_D')

        print('UP_D', upsample_block_D.get_shape().as_list())

        # (8, 16, 16)
        upsample_block_C = self.upsampling_block(
            upsample_block_D,
            block_C,
            filters=128,
            scope='UpsamplingBlock_C')

        print('UP_C', upsample_block_C.get_shape().as_list())

        # (16, 32, 32)
        upsample_block_B = self.upsampling_block(
            upsample_block_C,
            block_B,
            filters=64,
            scope='UpsamplingBlock_B')

        print('UP_B', upsample_block_B.get_shape().as_list())

        # (32, 64, 64)
        upsample_block_A = self.upsampling_block(
            upsample_block_B,
            block_A,
            filters=32,
            scope='UpsamplingBlock_A')

        print('UP_A', upsample_block_A.get_shape().as_list())

        y_pred = self.bn_conv3d(
            upsample_block_A,
            1, (1, 1, 1),
            name='final_conv',
            activation=tf.nn.sigmoid,
            padding='same')

        print("Output shape is: ", y_true.get_shape().as_list())

        return input_tensor, y_true, y_pred

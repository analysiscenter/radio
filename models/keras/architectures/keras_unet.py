# pylint: disable=not-context-manager
# pylint: disable=too-many-statements
""" Contains KerasUnet model class. """

from functools import wraps
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import (Input,
                          concatenate,
                          Conv3D,
                          MaxPooling3D,
                          UpSampling3D,
                          Activation)
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from ..keras_model import KerasModel


class KerasUnet(KerasModel):
    """ KerasUnet model for 3D scans implemented in keras. """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def bottleneck_block(self, input_tensor, filters, scope, padding='same'):
        """ Apply bottleneck block transform to input tensor. """
        with tf.variable_scope(scope):
            conv1 = Conv3D(filters, (3, 3, 3),
                           data_format='channels_first',
                           padding=padding)(input_tensor)
            conv1 = BatchNormalization(axis=1, momentum=0.1,
                                       scale=True)(conv1)
            conv1 = Activation('relu')(conv1)

            conv2 = Conv3D(filters, (3, 3, 3),
                           data_format='channels_first',
                           padding=padding)(conv1)
            conv2 = BatchNormalization(axis=1, momentum=0.1,
                                       scale=True)(conv2)
            conv2 = Activation('relu')(conv2)
        return conv2

    def reduction_block(self, input_tensor, filters, scope, pool_size=(2, 2, 2), padding='same'):
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
        - pool_size: tuple(int, int, int), size of pooling kernel along three axis;
        - padding: str, padding mode for convolutions, can be 'same' or 'valid';

        Returns:
        - ouput tensor, keras tensor;
        """
        with tf.variable_scope(scope):
            conv1 = Conv3D(filters, (3, 3, 3),
                           data_format='channels_first',
                           padding=padding)(input_tensor)
            conv1 = BatchNormalization(axis=1, momentum=0.1,
                                       scale=True)(conv1)
            conv1 = Activation('relu')(conv1)

            conv2 = Conv3D(filters, (3, 3, 3),
                           data_format='channels_first',
                           padding=padding)(conv1)
            conv2 = BatchNormalization(axis=1, momentum=0.1,
                                       scale=True)(conv2)
            conv2 = Activation('relu')(conv2)

            max_pool = MaxPooling3D(data_format='channels_first',
                                    pool_size=pool_size)(conv2)
        return conv2, max_pool

    def upsampling_block(self, input_tensor, scip_connect_tensor, filters, scope, padding='same'):
        """ Apply upsampling transform to two input tensors.

        First of all, UpSampling3D transform is applied to input_tensor. Then output
        tensor of this operation is concatenated with scip_connect_tensor. After this
        two 3D-convolutions with batch normalization before 'relu' activation
        are applied.

        Args:
        - input_tensor: keras tensor, input tensor from previous layer;
        - scip_connect_tensor: keras tensor, input tensor from simmiliar
        layer from reduction branch of UNet;
        - filters: int, number of filters in convolutional layers;
        - scope: str, name of scope for this block;
        - padding: str, padding mode for convolutions, can be 'same' or 'valid';

        Returns:
        - output tensor, keras tensor;
        """
        with tf.variable_scope(scope):
            upsample_tensor = UpSampling3D(data_format="channels_first",
                                           size=(2, 2, 2))(input_tensor)
            upsample_tensor = concatenate([upsample_tensor, scip_connect_tensor], axis=1)

            conv1 = Conv3D(filters, (3, 3, 3),
                           data_format="channels_first",
                           padding="same")(upsample_tensor)
            conv1 = BatchNormalization(axis=1, momentum=0.1,
                                       scale=True)(conv1)
            conv1 = Activation('relu')(conv1)

            conv2 = Conv3D(filters, (3, 3, 3),
                           data_format="channels_first",
                           padding="same")(conv1)
            conv2 = BatchNormalization(axis=1, momentum=0.1,
                                       scale=True)(conv2)
            conv2 = Activation('relu')(conv2)
        return conv2

    def build_model(self, *args, **kwargs):
        """ Build 3D unet model implemented in keras. """
        input_tensor = Input((1, 32, 64, 64))

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

        # Final convolution
        final_conv = Conv3D(1, (1, 1, 1),
                            activation='sigmoid',
                            data_format="channels_first",
                            padding='same')(upsample_block_A)

        # Building keras model
        model = Model(inputs=input_tensor, outputs=final_conv, name='unet')
        return model

    @wraps(keras.models.Model.compile)
    def compile(self, optimizer='adam', loss=dice_coef_loss, **kwargs):
        """ Compile unet model. """
        super().compile(optimizer=optimizer, loss=loss)

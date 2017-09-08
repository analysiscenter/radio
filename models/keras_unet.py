# pylint: disable=too-many-statements
""" Contains KerasUnet model class. """

import numpy as np
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

from .keras_model import KerasModel


def dice_coef(y_true, y_pred):
    """ Dice coefficient required by keras model as a part of loss function.

    Args:
    - y_true: keras tensor with targets;
    - y_pred: keras tensor with predictions;

    Returns:
    - keras tensor with dice coefficient value;
    """
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    answer = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return answer

def dice_coef_np(y_true, y_pred, smooth=1e-6):
    """ Dice coefficient for two input arrays.

    Args:
    - y_true: numpy array containing target values;
    - y_pred: numpy array containing predicted values;

    Returns:
    - float, dice coefficient value;
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) +  \
    + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """ Dice loss function.

    Args:
    - y_true: keras tensor containing target values;
    - y_pred: keras tensor containing predicted values;

    Returns:
    - keras tensor containing dice loss;
    """
    answer = -dice_coef(y_true, y_pred)
    return answer


class KerasUnet(KerasModel):
    """ KerasUnet model for 3D scans implemented in keras. """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def reduction_block(self, input_tensor, filters, scope, pool_size=(2, 2, 2), padding='same'):
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
        return max_pool

    def upsampling_block(self, input_tensor, scip_connect_tensor, filters, scope, padding='same'):
        with tf.variable_scope(scope):
            upsample_tensor = UpSampling3D(data_format="channels_first",
                                           size=(2, 2, 2))(input_tensor)
            upsample_tensor = concatenate([upsample_tensor, scip_connet_tensor], axis=1)

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

    def build_unet(self):
        """ Build 3D unet model implemented in keras. """
        input_tensor = Input((1, 32, 64, 64))

        # Downsampling or reduction layers: ReductionBlock_A, ReductionBlock_B, ReductionBlock_C, ReductionBlock_D
        reduct_block_A = self.reduction_block(input_tensor, 32,
                                              scope='ReductionBlock_A')
        reduct_block_B = self.reduction_block(reduct_block_A, 64,
                                              scope='ReductionBlock_B')
        reduct_block_C = self.reduction_block(reduct_block_B, 128,
                                              scope='ReductionBlock_C')
        reduct_block_D = self.reduction_block(reduct_block_C, 256,
                                              scope='ReductionBlock_D')

        # Bottleneck layer
        bottleneck_block = self.reduction_block(reduct_block_D, 512, scope='BottleneckBlock')

        # Upsampling Layers: UpsamplingBlock_D, UpsamplingBlock_C, UpsamplingBlock_B, UpsamplingBlock_A
        upsample_block_D = self.upsampling_block(bottleneck_block, reduction_block_D,
                                                 256, scope='UpsamplingBlock_D')

        upsample_block_C = self.upsampling_block(upsample_block_D, reduction_block_C,
                                                 128, scope='UpsamplingBlock_C')

        upsample_block_B = self.upsampling_block(upsample_block_C, reduction_block_B,
                                                 64, scope='UpsamplingBlock_B')

        upsample_block_A = self.upsampling_block(upsample_block_B, reduction_block_A,
                                                 32, scope='UpsamplingBlock_A')

        # Final convolution
        final_conv = Conv3D(1, (1, 1, 1),
                            activation='sigmoid',
                            data_format="channels_first",
                            padding='same')(upsample_block_A)

        # Building keras model
        model = Model(inputs=input_tensor, outputs=final_conv, name='unet')
        return model

    @classmethod
    def initialize_model(cls, *args, **kwargs):
        """ Initialize unet mode. """
        return cls.build_unet()

    def load_model(self, path):
        """ Load weights and description of keras model. """
        self.model = keras.models.load_model(path, custom_objects={'dice_coef':dice_coef,
                                                                   'dice_coef_loss':dice_coef_loss})

""" Contains KerasUnet model class. """

import numpy as np
import keras
from keras.models import Model
from keras.models import load_model
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

    @staticmethod
    def build_unet():
        """ Build 3D unet implemented in keras. """
        inputs = Input((1, 32, 64, 64))
        conv1 = Conv3D(32, (3, 3, 3), data_format="channels_first", padding="same")(inputs)
        conv1 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv1)
        conv1 = Activation("relu")(conv1)
        #conv1 = Dropout(0.3)(conv1)
        conv1 = Conv3D(32, (3, 3, 3), data_format="channels_first", padding="same")(conv1)
        conv1 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv1)
        conv1 = Activation("relu")(conv1)
        #conv1 = Dropout(0.3)(conv1)
        pool1 = MaxPooling3D(data_format="channels_first", pool_size=(2, 2, 2))(conv1)

        #conv2 = Dropout(0.3)(pool1)
        conv2 = Conv3D(64, (3, 3, 3), data_format="channels_first", padding="same")(pool1)
        conv2 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv2)
        conv2 = Activation("relu")(conv2)
        #conv2 = Dropout(0.3)(conv2)
        conv2 = Conv3D(64, (3, 3, 3), data_format="channels_first", padding="same")(conv2)
        conv2 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling3D(data_format="channels_first", pool_size=(2, 2, 2))(conv2)

        #conv3 = Dropout(0.3)(pool2)
        conv3 = Conv3D(128, (3, 3, 3), data_format="channels_first", padding="same")(pool2)
        conv3 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv3D(128, (3, 3, 3), data_format="channels_first", padding="same")(conv3)
        conv3 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv3)
        conv3 = Activation('relu')(conv3)
        pool3 = MaxPooling3D(data_format="channels_first", pool_size=(2, 2, 2))(conv3)

        #conv4 = Dropout(0.3)(pool3)
        conv4 = Conv3D(256, (3, 3, 3), data_format="channels_first", padding="same")(pool3)
        conv4 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = Conv3D(256, (3, 3, 3), data_format="channels_first", padding="same")(conv4)
        conv4 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv4)
        conv4 = Activation('relu')(conv4)
        pool4 = MaxPooling3D(data_format="channels_first", pool_size=(2, 2, 2))(conv4)

        #conv5 = Dropout(0.3)(pool4)
        conv5 = Conv3D(512, (3, 3, 3), data_format="channels_first", padding="same")(pool4)
        conv5 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = Conv3D(512, (3, 3, 3), data_format="channels_first", padding="same")(conv5)
        conv5 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv5)
        conv5 = Activation('relu')(conv5)

        up6 = concatenate([UpSampling3D(data_format="channels_first", size=(2, 2, 2))(conv5), conv4], axis=1)
        conv6 = Conv3D(256, (3, 3, 3), data_format="channels_first", padding="same")(up6)
        conv6 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv6)
        conv6 = Activation('relu')(conv6)
        conv6 = Conv3D(256, (3, 3, 3), data_format="channels_first", padding="same")(conv6)
        conv6 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv6)
        conv6 = Activation('relu')(conv6)

        up7 = concatenate([UpSampling3D(data_format="channels_first", size=(2, 2, 2))(conv6), conv3], axis=1)
        conv7 = Conv3D(128, (3, 3, 3), data_format="channels_first", padding="same")(up7)
        conv7 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv7)
        conv7 = Activation('relu')(conv7)
        conv7 = Conv3D(128, (3, 3, 3), data_format="channels_first", padding="same")(conv7)
        conv7 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv7)
        conv7 = Activation('relu')(conv7)

        up8 = concatenate([UpSampling3D(data_format="channels_first", size=(2, 2, 2))(conv7), conv2], axis=1)
        conv8 = Conv3D(64, (3, 3, 3), data_format="channels_first", padding="same")(up8)
        conv8 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv8)
        conv8 = Activation('relu')(conv8)
        conv8 = Conv3D(64, (3, 3, 3), data_format="channels_first", padding="same")(conv8)
        conv8 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv8)
        conv8 = Activation('relu')(conv8)

        up9 = concatenate([UpSampling3D(data_format="channels_first", size=(2, 2, 2))(conv8), conv1], axis=1)
        conv9 = Conv3D(32, (3, 3, 3), data_format="channels_first", padding="same")(up9)
        conv9 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv9)
        conv9 = Activation('relu')(conv9)
        conv9 = Conv3D(32, (3, 3, 3), data_format="channels_first", padding="same")(conv9)
        conv9 = BatchNormalization(axis=1, momentum=0.1, scale=True)(conv9)
        conv9 = Activation('relu')(conv9)
        conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid', data_format="channels_first", padding='same')(conv9)

        model = Model(inputs=inputs, outputs=conv10, name='unet')

        return model

    @classmethod
    def initialize_model(cls):
        """ Initialize unet mode. """
        return cls.build_unet()

    def load_model(self, path):
        """ Load weights and description of keras model. """
        self.log.info("Loading %s model..." % self.name)
        self.model = keras.models.load_model(path, custom_objects={'dice_coef':dice_coef,
                                                                   'dice_coef_loss':dice_coef_loss})
        self.log.info("Loaded model from %s" % path)

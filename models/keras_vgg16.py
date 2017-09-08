""" Contains implementation of VGG16 architecture in keras. """

from functools import wraps
import keras
from keras.models import Model
from keras.layers import Input, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import Dropout
from keras.layers import Dense, BatchNormalization

from .keras_model import KerasModel


def reduction_block_I(input_tensor, filters, scope, padding='same'):
    with tf.variable_scope(scope):
        conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                       activation='relu', padding=padding,
                       name='conv_1')(input_tensor)
        conv1 = BatchNormalization(axis=4)(conv1)

        conv2 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                       activaton='relu', padding=padding,
                       name='conv2')(conv1)
        conv2 = BatchNormalization(axis=4)(conv2)

        max_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2),
                                name='max_pool_3D')(conv2)
    return max_pool


def reduction_block_II(input_tensor, filters, scope, padding='same'):
    with tf.variable_scope(scope):
        conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                       activation='relu', padding=padding,
                       name='conv1')(input_tensor)
        conv1 = BatchNormalization(axis=4)(conv1)

        conv2 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                       activaton='relu', padding=padding,
                       name='conv2')(conv1)
        conv2 = BatchNormalization(axis=4)(conv2)

        conv3 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                       activaton='relu', padding=padding,
                       name='conv3')(conv2)
        conv3 = BatchNormalization(axis=4)(conv3)

        max_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2),
                                name='max_pool_3D')(conv3)
    return max_pool


def classification_block(input_tensor, filters=(512, 256), scope='ClassificationBlock'):
    with tf.variable_scope(scope):
        filters_1, filters_2 = filters

        layer = Flatten(name='flatten')(input_tensor)
        layer = Dense(filters_1, activation='relu', name='fc1')(layer)
        layer = BatchNormalization(axis=-1)(layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(filters_2, activation='relu', name='fc2')(layer)
        layer = BatchNormalization(axis=-1)(layer)
        layer = Dropout(0.2)(layer)
    return layer


class KerasVGG16(KerasModel):
    """ KerasVGG16 model for 3D scans implemented in keras. """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    @staticmethod
    def build_vgg16():
        input_tensor = Input(shape=(32, 64, 64, 64, 1))
        block_A = reduction_block_I(img_input, 32, scope='Block_A')
        block_B = reduction_block_I(block_A, 64, scope='Block_B')
        block_C = reduction_block_II(block_B, 128, scope='Block_C')
        block_D = reduction_block_II(block_C, 256, scope='Block_D')
        block_E = reduction_block_II(block_D, 256, scope='Block_E')

        block_F = classification_block(block_E, (512, 256),
                                       scope='ClassificationBlock')

        output_tensor = Dense(1, activation='sigmoid',
                              name='predictions')(block_F)

        model = Model(input_tensor, output_tensor, name='vgg16')
        return model

    @classmethod
    def initialize_model(cls):
        """ Initialize vgg16 model. """
        return cls.build_vgg16()

    @wraps(keras.models.Model.compile)
    def compile(self, optimizer='adam', loss='binary_crossentropy', **kwargs):
        """ Compile vgg16 model. """
        super().compile(optimizer=optimizer, loss=loss)

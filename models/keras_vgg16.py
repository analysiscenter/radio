""" Contains implementation of VGG16 architecture in keras. """

from functools import wraps
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import Dropout
from keras.layers import Flatten, Input, Activation, Dense, BatchNormalization

from .keras_model import KerasModel


class KerasVGG16(KerasModel):
    """ KerasVGG16 model for 3D scans implemented in keras. """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    @staticmethod
    def build_vgg16():
        """ Build vgg16 model for 3D scans implemented in keras. """

        img_input = Input(shape=(32, 64, 64, 1))
        layer = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu',
                       padding='same', name='block1_conv1')(img_input)
        layer = BatchNormalization(axis=4)(layer)

        layer = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu',
                       padding='same', name='block1_conv2')(layer)
        layer = BatchNormalization(axis=4)(layer)
        layer = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(layer)

        # Block 2
        layer = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu',
                       padding='same', name='block2_conv1')(layer)
        layer = BatchNormalization(axis=4)(layer)
        layer = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu',
                       padding='same', name='block2_conv2')(layer)
        layer = BatchNormalization(axis=4)(layer)
        layer = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(layer)

        # Block 3
        layer = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu',
                       padding='same', name='block3_conv1')(layer)
        layer = BatchNormalization(axis=4)(layer)
        layer = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu',
                       padding='same', name='block3_conv2')(layer)
        layer = BatchNormalization(axis=4)(layer)
        layer = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu',
                       padding='same', name='block3_conv3')(layer)
        layer = BatchNormalization(axis=4)(layer)
        layer = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block3_pool')(layer)

        # Block 4
        layer = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu',
                       padding='same', name='block4_conv1')(layer)
        layer = BatchNormalization(axis=4)(layer)
        layer = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu',
                       padding='same', name='block4_conv2')(layer)
        layer = BatchNormalization(axis=4)(layer)
        layer = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu',
                       padding='same', name='block4_conv3')(layer)
        layer = BatchNormalization(axis=4)(layer)
        layer = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool')(layer)

        # Block 5
        layer = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu',
                       padding='same', name='block5_conv1')(layer)
        layer = BatchNormalization(axis=4)(layer)
        layer = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu',
                       padding='same', name='block5_conv2')(layer)
        layer = BatchNormalization(axis=4)(layer)
        layer = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu',
                       padding='same', name='block5_conv3')(layer)
        layer = BatchNormalization(axis=4)(layer)
        layer = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block5_pool')(layer)

        # Classification block
        layer = Flatten(name='flatten')(layer)
        layer = Dense(512, activation='relu', name='fc1')(layer)
        layer = BatchNormalization(axis=-1)(layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(256, activation='relu', name='fc2')(layer)
        layer = BatchNormalization(axis=-1)(layer)
        layer = Dropout(0.2)(layer)

        layer = Dense(1, activation='sigmoid', name='predictions')(layer)

        model = Model(img_input, layer, name='vgg16')
        return model

    @classmethod
    def initialize_model(cls):
        """ Initialize vgg16 model. """
        return cls.build_vgg16()

    @wraps(keras.models.Model.compile)
    def compile(self, optimizer='adam', loss='binary_crossentropy', **kwargs):
        """ Compile vgg16 model. """
        super().compile(optimizer=optimizer, loss=loss)

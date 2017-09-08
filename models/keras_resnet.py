""" Contains implementation of ResNet via keras. """

from keras import layers
from keras.models import Model

from keras.layers import Input
from keras.layers import Dense, Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv3D
from keras.layers import BatchNormalization

from .keras_model import KerasModel


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """ The identity block is the block that has no conv layer at shortcut. """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(axis=4, name=bn_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = Conv3D(filters1, (1, 1, 1), name=conv_name_base + '2a',
               use_bias=False, kernel_initializer='glorot_normal')(x)

    x = BatchNormalization(axis=4, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv3D(filters2, kernel_size,
               padding='same',
               name=conv_name_base + '2b',
               use_bias=False,
               kernel_initializer='glorot_normal')(x)

    x = BatchNormalization(axis=4, name=bn_name_base + '2c')(x)
    x = Conv3D(filters3, (1, 1, 1),
               name=conv_name_base + '2c',
               use_bias=False,
               kernel_initializer='glorot_normal')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2, 2)):
    """ A block that has a conv layer at shortcut. """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(axis=4, name=bn_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = Conv3D(filters1, (1, 1, 1),
               strides=strides,
               name=conv_name_base + '2a',
               use_bias=False,
               kernel_initializer='glorot_normal')(x)

    x = BatchNormalization(axis=4, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv3D(filters2,
               kernel_size,
               padding='same',
               name=conv_name_base + '2b',
               use_bias=False,
               kernel_initializer='glorot_normal')(x)

    x = BatchNormalization(axis=4, name=bn_name_base + '2c')(x)
    x = Conv3D(filters3, (1, 1, 1),
               name=conv_name_base + '2c',
               use_bias=False,
               kernel_initializer='glorot_normal')(x)

    shortcut = BatchNormalization(axis=4, name=bn_name_base + '1')(input_tensor)
    shortcut = Conv3D(filters3, (1, 1, 1),
                      strides=strides,
                      name=conv_name_base + '1',
                      use_bias=False,
                      kernel_initializer='glorot_normal')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


class KerasResNet(KerasModel):
    """ ResNet model for 3D scans implemented in keras. """

    def __init__(self, name, dropout_rate=0.3, **kwargs):
        self.dropout_rate = dropout_rate
        self.input_tensor = Input(shape=(32, 64, 64, 1))
        super().__init__(name, dropout_rate=dropout_rate, input_tensor=self.input_tensor, **kwargs)

    @staticmethod
    def build_resnet(input_tensor, dropout_rate):
        """ Build resnet model implemented in keras.

        Args:
        - input_tensor: keras Input layer;
        - dropout_rate: float, dropout_rate for dense layers;

        Returns:
        - keras tensor of the last dense layer;
        """
        x = Conv3D(filters=32, kernel_size=(7, 3, 3),
                   strides=(2, 2, 2), name='initial_conv', padding='same',
                   use_bias=False, kernel_initializer='glorot_normal')(input_tensor)
        x = BatchNormalization(axis=4, name='initial_batch_norm')(x)
        x = Activation('relu')(x)

        x = conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1, 1))
        x = identity_block(x, 3, [16, 16, 64], stage=2, block='b')
        x = identity_block(x, 3, [16, 16, 64], stage=2, block='c')

        x = conv_block(x, 3, [32, 32, 128], stage=3, block='a')
        x = identity_block(x, 3, [32, 32, 128], stage=3, block='b')
        x = identity_block(x, 3, [32, 32, 128], stage=3, block='c')
        x = identity_block(x, 3, [32, 32, 128], stage=3, block='d')

        x = conv_block(x, 3, [64, 64, 256], stage=4, block='a')
        x = identity_block(x, 3, [64, 64, 256], stage=4, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=4, block='c')
        x = identity_block(x, 3, [64, 64, 256], stage=4, block='d')
        x = identity_block(x, 3, [64, 64, 256], stage=4, block='e')
        x = identity_block(x, 3, [64, 64, 256], stage=4, block='f')

        x = conv_block(x, 3, [128, 128, 512], stage=5, block='a', strides=(1, 2, 2))
        x = identity_block(x, 3, [128, 128, 512], stage=5, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=5, block='c')

        y = Flatten()(x)

        y = Dense(512, activation='relu')(y)
        y = Dropout(rate=dropout_rate)(y)

        y = BatchNormalization(axis=-1)(y)
        y = Dense(32, activation='relu')(y)

        output_layer = Dense(1, activation='sigmoid', name='output')(y)

        model = Model(input_tensor, output_layer, name='resnet')
        return model

    @classmethod
    def initialize_model(cls, dropout_rate, input_tensor, *args, **kwargs):
        """ Initialize ResNet model. """
        # resnet_model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        return cls.build_resnet(input_tensor, dropout_rate)

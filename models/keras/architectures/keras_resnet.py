""" Contains implementation of ResNet via keras. """

from keras import layers
from keras.models import Model

from keras.layers import Input
from keras.layers import Dense, Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv3D
from keras.layers import BatchNormalization

from ..keras_model import KerasModel


class KerasResNet50(KerasModel):
    """ ResNet50 model for 3D scans implemented in keras. """

    def __init__(self, *args, **kwargs):
        """ Call __init__ of KerasModel. """
        super().__init__(*args, **kwargs)

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """ The identity block is the block that has no conv layer at shortcut.

        This block consists of two convolutions with batch normalization before
        'relu' activation. After three convolutions are applyed the ouput tensor
        is concatenated with input tensor along filters dimension and go
        through 'relu' activation.

        Args:
        - input_tensor: keras tensor, input tensor;
        - kernel_size: tuple(int, int, int), size of the kernel along three dimensions
        for all convolution operations in block.
        - filters: tuple(int, int, int), number of filters in first, second and
        third 3D-convolution operations;
        - stage: int, number of stage, on par with block argument
        used to derive names of inner layers;
        - block: str, block prefix, on par with stage argument used
        to derive names of inner layers;

        Returns:
        - output tensor, keras tensor;
        """
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv3D(filters1, (1, 1, 1), name=conv_name_base + '2a',
                   use_bias=False, kernel_initializer='glorot_normal')(input_tensor)
        x = BatchNormalization(axis=4, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv3D(filters2, kernel_size,
                   padding='same',
                   name=conv_name_base + '2b',
                   use_bias=False,
                   kernel_initializer='glorot_normal')(x)
        x = BatchNormalization(axis=4, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv3D(filters3, (1, 1, 1),
                   name=conv_name_base + '2c',
                   use_bias=False,
                   kernel_initializer='glorot_normal')(x)
        x = BatchNormalization(axis=4, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2, 2)):
        """ A block that has a conv layer at shortcut.

        This layer consists of three 3D-convolutional layers with batch
        normalization before 'relu' activation. The output tesnor then
        concatenated with result tensor of 3D-convolution applied to input tensor
        of the block with (2, 2, 2)-strides.

        Args:
        - input_tensor: keras tensor, input tensor;
        - kernel_size: tuple(int, int, int), size of the kernel along three dimensions
        for all convolution operations in block.
        - filters: tuple(int, int, int), number of filters in first, second and
        third 3D-convolution operations;
        - stage: int, number of stage, on par with block argument
        used to derive names of inner layers;
        - block: str, block prefix, on par with stage argument used
        to derive names of inner layers;

        Returns:
        - output tensor, keras tensor;
        """
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = BatchNormalization(axis=4, name=bn_name_base + '2a')(input_tensor)
        x = Conv3D(filters1, (1, 1, 1),
                   strides=strides,
                   name=conv_name_base + '2a',
                   use_bias=False,
                   kernel_initializer='glorot_normal')(x)
        x = Activation('relu')(x)

        x = BatchNormalization(axis=4, name=bn_name_base + '2b')(x)
        x = Conv3D(filters2,
                   kernel_size,
                   padding='same',
                   name=conv_name_base + '2b',
                   use_bias=False,
                   kernel_initializer='glorot_normal')(x)
        x = Activation('relu')(x)

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

    def _build(self, dropout_rate=0.3, units=(256, 64), *args, **kwargs):
        """ Build resnet50 model implemented in keras.

        Args:
        - input_tensor: keras Input layer;
        - dropout_rate: float, dropout_rate for dense layers;

        Returns:
        - tuple([*input_nodes], [*output_nodes]);
        """
        units_1, units_2 = units

        input_tensor = Input(shape=(32, 64, 64, 1))
        x = Conv3D(filters=32, kernel_size=(7, 3, 3),
                   strides=(2, 2, 2), name='initial_conv', padding='same',
                   use_bias=False, kernel_initializer='glorot_normal')(input_tensor)

        x = BatchNormalization(axis=4, name='initial_batch_norm')(x)
        x = Activation('relu')(x)

        x = self.conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1, 1))
        x = self.identity_block(x, 3, [16, 16, 64], stage=2, block='b')
        x = self.identity_block(x, 3, [16, 16, 64], stage=2, block='c')

        x = self.conv_block(x, 3, [32, 32, 128], stage=3, block='a')
        x = self.identity_block(x, 3, [32, 32, 128], stage=3, block='b')
        x = self.identity_block(x, 3, [32, 32, 128], stage=3, block='c')
        x = self.identity_block(x, 3, [32, 32, 128], stage=3, block='d')

        x = self.conv_block(x, 3, [64, 64, 256], stage=4, block='a')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='c')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='d')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='e')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='f')

        x = self.conv_block(x, 3, [128, 128, 512], stage=5, block='a', strides=(2, 2, 2))
        x = self.identity_block(x, 3, [128, 128, 512], stage=5, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=5, block='c')

        y = Flatten()(x)

        y = Dense(units_1, activation='relu')(y)
        y = Dropout(rate=dropout_rate)(y)

        y = BatchNormalization(axis=-1)(y)
        y = Dense(units_2, activation='relu')(y)

        output_layer = Dense(1, activation='sigmoid', name='output')(y)

        return [input_tensor], [output_layer]

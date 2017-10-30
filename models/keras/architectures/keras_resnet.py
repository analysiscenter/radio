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
    """ ResNet50 model for 3D scans implemented in keras.

    This class extends KerasModel class.

    Contains description of three types of blocks:
    'identity_block' and 'conv_block'. ResNet architercture is implemented inside
    _build method using these blocks. Full description of similar 2D model
    architecture can be downloaded from here: https://arxiv.org/pdf/1512.03385v1.pdf

    Attributes
    ----------
    config : dict
        config dictionary from dataset pipeline
        see configuring model section of dataset module
        https://github.com/analysiscenter/dataset/blob/models/doc/models.md#configuring-a-model.
    name : str
        name of the model.
    units : tuple(int, int)
        number of units in two final dense layers before tensor with predicitons.
    num_targets : int
        size of tensor with predicitons.
    dropout_rate : float
        probability of dropout.
    """

    def __init__(self, *args, **kwargs):
        """ Call __init__ of KerasModel and add specific for KerasResNet50 attributes. """
        self.config = kwargs.get('config', {})
        self.dropout_rate = self.get_from_config('dropout_rate', 0.35)
        self.num_targets = self.get_from_config('num_targets', 1)
        self.units = self.get_from_config('units', (256, 128))
        super().__init__(*args, **kwargs)

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """ The identity block is the block that has no conv layer at shortcut.

        Parameters
        ----------
        input_tensor : keras tensor
            input tensor.
        kernel_size : tuple(int, int, int)
            size of the kernel along three dimensions for middle convolution operation in block.
        filters : tuple(int, int, int)
            number of filters in first, second and third 3D-convolution operations.
        stage : int
            number of stage, on par with block argument used to derive names of inner layers.
        block : str
            block prefix, on par with stage argument used to derive names of inner layers.

        Returns
        -------
        keras tensor
            output tensor.
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

        Parameters
        ----------
        input_tensor : keras tensor
            input tensor.
        kernel_size : tuple(int, int, int)
            size of the kernel along three dimensions for middle convolution operation in block.
        filters : tuple(int, int, int)
            number of filters in first, second and third 3D-convolution operations.
        stage : int
            number of stage, on par with block argument used to derive names of inner layers.
        block : str
            block prefix, on par with stage argument used to derive names of inner layers.

        Returns
        -------
        keras tensor
            output tensor.
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

    def _build(self, *args, **kwargs):
        """ Build resnet50 model implemented in keras.

        Returns
        -------
        tuple([*input_nodes], [*output_nodes]);
            list of input nodes and list of output nodes.
        """
        units_1, units_2 = self.units

        input_tensor = Input(shape=(32, 64, 64, 1))
        x = Conv3D(filters=32, kernel_size=(5, 3, 3),
                   strides=(1, 2, 2), name='initial_conv', padding='same',
                   use_bias=False, kernel_initializer='glorot_normal')(input_tensor)

        x = BatchNormalization(axis=4, name='initial_batch_norm')(x)
        x = Activation('relu')(x)

        x = self.conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1, 1))
        x = self.identity_block(x, 3, [16, 16, 64], stage=2, block='b')
        x = self.identity_block(x, 3, [16, 16, 64], stage=2, block='c')
        x = Dropout(rate=self.dropout_rate)(x)

        x = self.conv_block(x, 3, [32, 32, 128], stage=3, block='a')
        x = self.identity_block(x, 3, [32, 32, 128], stage=3, block='b')
        x = self.identity_block(x, 3, [32, 32, 128], stage=3, block='c')
        x = self.identity_block(x, 3, [32, 32, 128], stage=3, block='d')
        x = Dropout(rate=self.dropout_rate)(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=4, block='a')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='c')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='d')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='e')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='f')
        x = Dropout(rate=self.dropout_rate)(x)

        x = self.conv_block(x, 3, [128, 128, 512], stage=5, block='a', strides=(2, 2, 2))
        x = self.identity_block(x, 3, [128, 128, 512], stage=5, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=5, block='c')

        y = Flatten()(x)

        y = Dense(units_1, activation='relu')(y)
        y = Dropout(rate=self.dropout_rate)(y)

        y = BatchNormalization(axis=-1)(y)
        y = Dense(units_2, activation='relu')(y)

        output_layer = Dense(self.num_targets, activation='sigmoid', name='output')(y)

        return [input_tensor], [output_layer]

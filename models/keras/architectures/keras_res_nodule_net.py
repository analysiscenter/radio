# pylint: disable=anomalous-backslash-in-string
""" Contains implementation of ResNet via keras. """

from keras import layers

from keras.layers import Input
from keras.layers import Dense, Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv3D
from keras.layers import BatchNormalization

from ..keras_model import KerasModel


class KerasResNoduleNet(KerasModel):
    """ ResNoduleNet model for 3D scans implemented in keras.

    Class extends KerasModel class.

    Contains description of three types of blocks:
    'identity_block' and 'conv_block'. ResNet architercture is implemented inside
    _build method using these blocks.
    Model is inspired by ResNet (Kaiming He et Al., https://arxiv.org/abs/1512.03385/).

    Attributes
    ----------
    config : dict
        config dictionary from dataset pipeline
        see configuring model section of dataset module:
        https://analysiscenter.github.io/dataset/intro/models.html.
    name : str
        name of the model.
    units : tuple(int, int) or int or None
        number of units in final dense layers before tensor with
        predicitons. default: (512, 256).
    num_targets : int
        size of tensor with predicitons. default: 1.
    dropout_rate : float
        probability of dropout. default: 0.35.

    Note
    ----
    Implementation requires the input tensor having shape=(batch_size, 32, 64, 64, 1).
    """

    def __init__(self, *args, **kwargs):
        """ Call __init__ of KerasModel and add specific for KerasResNet50 attributes. """
        self.config = kwargs.get('config', {})
        self.dropout_rate = self.get_from_config('dropout_rate', 0.35)
        self.num_targets = self.get_from_config('num_targets', 1)

        units = self.get_from_config('units', (512, 256))
        if units is None:
            units = ()
        elif isinstance(units, int):
            units = (units, )
        self.units = tuple(units)

        super().__init__(*args, **kwargs)

    def identity_block(self, inputs, kernel_size, filters, stage, block):
        """ The identity block is the block that has no conv layer at shortcut.

        First of all, 3D-convolution with (1, 1, 1) kernel size, batch normalization
        and relu activation is applied. Then the result flows into
        3D-convolution with (3, 3, 3) kernel size, batch normalization and
        relu activation. Finally, the result of previous convolution goes
        into 3D-convolution with (1, 1, 1) kernel size, batch normalization
        without activation and its output is summed with the input tensor
        and `relu` activation is applied.
        Argument `filters` should be tuple(int, int, int) and specifies
        number of filters in first, second and third convolution correspondingly.
        Number of filters in third convolution must be the same as in the input
        tensor.

        Parameters
        ----------
        inputs : keras tensor
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
                   use_bias=False, kernel_initializer='glorot_normal')(inputs)
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

        x = layers.add([x, inputs])
        x = Activation('relu')(x)
        return x

    def conv_block(self, inputs, kernel_size, filters, stage, block, strides=(2, 2, 2)):
        """ Convolutional block that has a conv layer at shortcut.

        3D-convolution with (1, 1, 1) kernel size, (2, 2, 2)-strides,
        batch normalization and `relu` activation are applied. Then resulting tensor
        flows into 3D-convolution with (3, 3, 3) kernel size, batch normalization
        and `relu` activation. Finally, the result of previous convolution goes
        into 3D-convolution with (1, 1, 1) kernel size, batch normalization
        without activation and its output is summed with the result
        of 3D-convolution with kernel_size=(1, 1, 1), strides=(2, 2, 2) and
        batch normalization of inputs. After that `relu` activation
        is applied to the result of `add` operation.
        Argument `filters` should be tuple(int, int, int) and specifies
        number of filters in first, second and third convolution correspondingly.

        Parameters
        ----------
        inputs : keras tensor
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

        x = Conv3D(filters1, (1, 1, 1),
                   strides=strides,
                   name=conv_name_base + '2a',
                   use_bias=False,
                   kernel_initializer='glorot_normal')(inputs)
        x = BatchNormalization(axis=4, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv3D(filters2,
                   kernel_size,
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

        shortcut = Conv3D(filters3, (1, 1, 1),
                          strides=strides,
                          name=conv_name_base + '1',
                          use_bias=False,
                          kernel_initializer='glorot_normal')(inputs)
        shortcut = BatchNormalization(
            axis=4, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def _build(self, *args, **kwargs):
        """ Build ResNoduleNet model implemented in keras.

        Returns
        -------
        tuple([*input_nodes], [*output_nodes]);
            list of input nodes and list of output nodes.
        """
        num_targets = self.get('num_targets', self.config)
        dropout_rate = self.get('dropout_rate', self.config)
        input_shape = self.get('input_shape', self.config)

        inputs = Input(shape=input_shape)
        x = Conv3D(filters=32, kernel_size=(5, 3, 3),
                   strides=(1, 2, 2), name='initial_conv', padding='same',
                   use_bias=False, kernel_initializer='glorot_normal')(inputs)

        x = BatchNormalization(axis=4, name='initial_batch_norm')(x)
        x = Activation('relu')(x)

        x = self.conv_block(x, 3, [16, 16, 64], stage=2,
                            block='a', strides=(1, 1, 1))
        x = self.identity_block(x, 3, [16, 16, 64], stage=2, block='b')
        x = self.identity_block(x, 3, [16, 16, 64], stage=2, block='c')
        x = Dropout(rate=dropout_rate)(x)

        x = self.conv_block(x, 3, [32, 32, 128], stage=3, block='a')
        x = self.identity_block(x, 3, [32, 32, 128], stage=3, block='b')
        x = self.identity_block(x, 3, [32, 32, 128], stage=3, block='c')
        x = self.identity_block(x, 3, [32, 32, 128], stage=3, block='d')
        x = Dropout(rate=dropout_rate)(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=4, block='a')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='c')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='d')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='e')
        x = self.identity_block(x, 3, [64, 64, 256], stage=4, block='f')
        x = Dropout(rate=dropout_rate)(x)

        x = self.conv_block(x, 3, [128, 128, 512], stage=5, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=5, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=5, block='c')

        z = self.dense_block(x, units=self.get('units', self.config),
                             dropout=False, scope='DenseBlock-I')

        output_layer = Dense(
            self.num_targets, activation='sigmoid', name='output')(z)

        return [inputs], [output_layer]

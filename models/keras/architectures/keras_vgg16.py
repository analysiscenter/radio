# pylint: disable=not-context-manager
""" Contains implementation of VGG16 architecture in keras. """

import tensorflow as tf
from keras.layers import Input, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import Dropout
from keras.layers import Dense, BatchNormalization

from ..keras_model import KerasModel


class KerasVGG16(KerasModel):
    """ KerasVGG16 model for 3D scans implemented in keras.

    This class extends KerasModel class.

    Contains description of three types of blocks:
    'reduction_block_I', 'reduction_block_II' and 'classification_block'.
    VGG16 architecture is implemented inside _build method using these blocks.

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
        """ Call __init__ of KerasModel and add specific for KerasVGG16 attributes. """
        self.config = kwargs.get('config', {})
        self.units = self.get_from_config('units', (512, 256))
        self.dropout_rate = self.get_from_config('dropout_rate', 0.35)
        self.num_targets = self.get_from_config('num_targets', 1)
        super().__init__(*args, **kwargs)

    def reduction_block_I(self, input_tensor, filters, scope, padding='same'):
        """ Reduction block of type I for VGG16 architecture.

        Applyes 3D-convolution with kernel size (3, 3, 3), (1, 1, 1) strides
        and 'relu' activation, after performs batch noramlization, then
        again 3D-convolution with kernel size (3, 3, 3),
        strides (1, 1, 1) and 'relu' activation,  that batch normalization;
        After all applyes 3D maxpooling operation with strides (2, 2, 2)
        and pooling size (2, 2, 2).

        Parameters
        ----------
        input_tensor : keras tensor
            input tensor.
        filters : int
            number of filters in 3D-convolutional layers.
        scope : str
            scope name for this block, will be used as an argument of tf.variable_scope.
        padding : str
            padding mode can be 'same' or 'valid'.

        Returns
        -------
        keras tensor
            output tensor.
        """
        with tf.variable_scope(scope):
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           activation='relu', padding=padding)(input_tensor)

            conv1 = BatchNormalization(axis=4)(conv1)

            conv2 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           activation='relu', padding=padding)(conv1)
            conv2 = BatchNormalization(axis=4)(conv2)

            max_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv2)
        return max_pool

    def reduction_block_II(self, input_tensor, filters, scope, padding='same'):
        """ Reduction block of type II for VGG16 architecture.

        Applyes 3D-convolution with kernel size (3, 3, 3), strides (1, 1, 1)
        and 'relu' activation, after that preform batch noramlization,
        repeates this combo three times;
        Finally, adds 3D maxpooling layer with
        strides (2, 2, 2) and pooling size (2, 2, 2).

        Parameters
        ----------
        input_tensor : keras tensor
            input tensor.
        filters : int
            number of filters in 3D-convolutional layers.
        scope : str
            scope name for this block, will be used as an argument of tf.variable_scope.
        padding : str
            padding mode can be 'same' or 'valid'.

        Returns
        -------
        keras tensor
            output tensor.
        """
        with tf.variable_scope(scope):
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           activation='relu', padding=padding)(input_tensor)
            conv1 = BatchNormalization(axis=4)(conv1)

            conv2 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           activation='relu', padding=padding)(conv1)
            conv2 = BatchNormalization(axis=4)(conv2)

            conv3 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           activation='relu', padding=padding)(conv2)
            conv3 = BatchNormalization(axis=4)(conv3)

            max_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv3)
        return max_pool

    def classification_block(self, input_tensor, scope='ClassificationBlock'):
        """ Classification block of VGG16 architecture.

        This block consists of flatten operation applied to input_tensor.
        Then there is two fully connected layers with 'relu' activation,
        batch normalization and dropout layers. This block should be put
        in the end of the model.

        Parameters
        ----------
        input_tensor : keras tensor
            input tensor.
        units : tuple(int, int)
            number of units in first and second dense layers.
        dropoout_rate : float
            probability of dropout.
        scope : str
            scope name for this block, will be used as an argument of tf.variable_scope.

        Returns:
        keras tensor
            output tensor.
        """
        with tf.variable_scope(scope):
            units_1, units_2 = self.units

            layer = Flatten(name='flatten')(input_tensor)
            layer = Dense(units_1, activation='relu', name='fc1')(layer)
            layer = BatchNormalization(axis=-1)(layer)
            layer = Dropout(dropout_rate)(layer)
            layer = Dense(units_2, activation='relu', name='fc2')(layer)
            layer = BatchNormalization(axis=-1)(layer)
            layer = Dropout(self.dropout_rate)(layer)
        return layer

    def _build(self, *args, **kwargs):
        """ Build VGG16 model implemented in keras.

        Returns
        -------
        tuple([*input_nodes], [*output_nodes])
            list of input nodes and list of output nodes.
        """
        input_tensor = Input(shape=(32, 64, 64, 1))
        block_A = self.reduction_block_I(input_tensor, 32, scope='Block_A')
        block_B = self.reduction_block_I(block_A, 64, scope='Block_B')
        block_C = self.reduction_block_II(block_B, 128, scope='Block_C')
        block_D = self.reduction_block_II(block_C, 256, scope='Block_D')
        block_E = self.reduction_block_II(block_D, 256, scope='Block_E')

        block_F = self.classification_block(block_E, scope='ClassificationBlock')

        output_tensor = Dense(self.num_targets,
                              activation='sigmoid',
                              name='predictions')(block_F)

        return [input_tensor], [output_tensor]

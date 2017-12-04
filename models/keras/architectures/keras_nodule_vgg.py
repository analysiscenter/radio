""" Contains implementation of VGG16 architecture in keras. """

import tensorflow as tf
from keras.layers import Input
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import Dense, BatchNormalization

from ..keras_model import KerasModel


class KerasNoduleVGG(KerasModel):
    """ KerasNoduleVGG model for 3D scans implemented in keras.

    Class extends KerasModel class.

    Contains description of three types of blocks:
    'reduction_block_I', 'reduction_block_II' and 'classification_block'.
    NoduleVGG architecture is implemented inside _build method using these blocks.

    Attributes
    ----------
    config : dict
        config dictionary from dataset pipeline
        see configuring model section of dataset module
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
    def reduction_block_I(self, inputs, filters, scope, padding='same'):
        """ Reduction block of type I for NoduleVGG architecture.

        Applyes 3D-convolution with kernel size (3, 3, 3), (1, 1, 1) strides
        and 'relu' activation, after performs batch noramlization, then
        again 3D-convolution with kernel size (3, 3, 3),
        strides (1, 1, 1) and 'relu' activation,  that batch normalization;
        After all applyes 3D maxpooling operation with strides (2, 2, 2)
        and pooling size (2, 2, 2).

        Parameters
        ----------
        inputs : keras tensor
            input tensor.
        filters : int
            number of filters in 3D-convolutional layers.
        scope : str
            scope name for block, will be used as an argument of tf.variable_scope.
        padding : str
            padding mode can be 'same' or 'valid'.

        Returns
        -------
        keras tensor
            output tensor.
        """
        with tf.variable_scope(scope):
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           activation='relu', padding=padding)(inputs)

            conv1 = BatchNormalization(axis=4)(conv1)

            conv2 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           activation='relu', padding=padding)(conv1)
            conv2 = BatchNormalization(axis=4)(conv2)

            max_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv2)
        return max_pool

    def reduction_block_II(self, inputs, filters, scope, padding='same'):
        """ Reduction block of type II for NoduleVGG architecture.

        Applyes 3D-convolution with kernel size (3, 3, 3), strides (1, 1, 1)
        and 'relu' activation, after that preform batch noramlization,
        repeates combo three times;
        Finally, adds 3D maxpooling layer with
        strides (2, 2, 2) and pooling size (2, 2, 2).

        Parameters
        ----------
        inputs : keras tensor
            input tensor.
        filters : int
            number of filters in 3D-convolutional layers.
        scope : str
            scope name for block, will be used as an argument of tf.variable_scope.
        padding : str
            padding mode can be 'same' or 'valid'.

        Returns
        -------
        keras tensor
            output tensor.
        """
        with tf.variable_scope(scope):
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           activation='relu', padding=padding)(inputs)
            conv1 = BatchNormalization(axis=4)(conv1)

            conv2 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           activation='relu', padding=padding)(conv1)
            conv2 = BatchNormalization(axis=4)(conv2)

            conv3 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           activation='relu', padding=padding)(conv2)
            conv3 = BatchNormalization(axis=4)(conv3)

            max_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv3)
        return max_pool

    def _build(self, *args, **kwargs):
        """ Build NoduleVGG model implemented in keras.

        Returns
        -------
        tuple([*input_nodes], [*output_nodes])
            list of input nodes and list of output nodes.
        """
        num_targets = self.get('num_targets', self.config)
        dropout_rate = self.get('dropout_rate', self.config)
        input_shape = self.get('input_shape', self.config)

        inputs = Input(shape=input_shape)
        block_A = self.reduction_block_I(inputs, 32, scope='Block_A')
        block_B = self.reduction_block_I(block_A, 64, scope='Block_B')
        block_C = self.reduction_block_II(block_B, 128, scope='Block_C')
        block_D = self.reduction_block_II(block_C, 256, scope='Block_D')
        block_E = self.reduction_block_II(block_D, 256, scope='Block_E')

        z = self.dense_block(block_E, units=self.get('units', self.config),
                             dropout=dropout_rate, scope='DenseBlock-I')

        output_tensor = Dense(num_targets, activation='sigmoid', name='output')(z)

        return [inputs], [output_tensor]

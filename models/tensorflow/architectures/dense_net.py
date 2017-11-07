# pylint: disable=too-many-arguments
# pylint: disable=not-context-manager
""" Contains DenseNet model class. """

import tensorflow as tf
from ..tf_model import TFModelCT
from ..layers import bn_conv3d, global_average_pool3d


class TFDenseNet(TFModelCT):
    """ This class implements 3D DenseNet architecture via tensorflow.

    Attributes
    ----------
    config : dict
        config dictionary from dataset pipeline
        see configuring model section of dataset module
        https://analysiscenter.github.io/dataset/intro/models.html.
    name : str
        name of the model.
    units : tuple(int, int)
        number of units in two final dense layers before tensor with predicitons.
    num_targets : int
        size of tensor with predicitons.
    dropout_rate : float
        probability of dropout.

    Full description of similar 2D model architecture can be downloaded from here:
    https://arxiv.org/pdf/1608.06993v2.pdf

    NOTE
    ----
    This class is descendant of TFModel class from dataset.models.*.
    """

    def __init__(self, *args, **kwargs):
        self.config = kwargs.get('config', {})
        self.num_targets = self.get_from_config('num_targets', 1)
        self.dropout_rate = self.get_from_config('dropout_rate', 0.35)
        super().__init__(*args, **kwargs)

    def dense_block(self, input_tensor, filters, block_size, name):
        """ Dense block which is used as a build block of densenet model.

        Schematically this layer can be represented like this:
        ==================================================================================
        input => conv3D{1x1x1}[1:1:1](filters) => conv3D{3x3x3}[1:1:1](filters) => output_1
        ----------------------------------------------------------------------------------
        concat([input, output_1]) => conv3D{1x1x1}[1:1:1](filters) =>
        => conv3D{3x3x3}[1:1:1](filters) => ouput_2
        ----------------------------------------------------------------------------------
        ...
        ----------------------------------------------------------------------------------
        ...
        ----------------------------------------------------------------------------------
        concat([input, output_1, ..., output_(n - 1)]) => conv3D{1x1x1}[1:1:1](filters) =>
        => conv3D{3x3x3}[1:1:1](filters) => output_n
        =================================================================================

        Parameters
        ----------
        input_tensor : tf.Tensor
            input_tensor.
        filters : int
            number of filters in output tensor of each convolution layer.
        block_size : int
            how many time repeat block conv3d{1x1x1} => conv3d{3x3x3} sequentially.
        name : str
            name of the layer that will be used as an argument of tf.variable_scope.

        Returns
        -------
        tf.Tensor
            output_tensor.
        """
        with tf.variable_scope(name):
            previous_input = tf.identity(input_tensor)
            for i in range(block_size):
                subblock_name = 'sub_block_' + str(i)
                x = bn_conv3d(previous_input, filters=filters,
                              kernel_size=(1, 1, 1),
                              strides=(1, 1, 1),
                              name=subblock_name + '_conv3d_1_1',
                              padding='same',
                              is_training=self.is_training)

                x = bn_conv3d(x, filters=filters,
                              kernel_size=(3, 3, 3),
                              strides=(1, 1, 1),
                              name=subblock_name + '_conv3d_3_3',
                              padding='same',
                              is_training=self.is_training)

                previous_input = tf.concat([previous_input, x], axis=-1)

        return previous_input

    def transition_layer(self, input_tensor, filters, name):
        """ Transition layer which is used as a dimension reduction block in densenset model.

        Parameters
        ----------
        input_tensor : tf.Tensor
            input tensor.
        filters : int
            number of filters in output tensor, required by 3D-convolution operation.
        name : str
            name of the layer that will be used as an argument of tf.variable_scope.

        Returns
        -------
        tf.Tensor
            output tensor.
        """
        with tf.variable_scope(name):
            output_tensor = bn_conv3d(input_tensor, filters=filters,
                                      kernel_size=(1, 1, 1),
                                      strides=(1, 1, 1),
                                      name='conv3d_1_1',
                                      is_training=self.is_training)

            output_tensor = tf.layers.average_pooling3d(output_tensor,
                                                        pool_size=(2, 2, 2),
                                                        strides=(2, 2, 2),
                                                        padding='same',
                                                        name='averagepool3d_2_2')
        return output_tensor

    def _build(self, *args, **kwargs):
        """ Build densenet model implemented via tensorflow. """
        input_tensor = tf.placeholder(shape=(None, 32, 64, 64, 1),
                                      dtype=tf.float32, name='input')

        y_true = tf.placeholder(shape=(None, self.num_targets), dtype=tf.float32, name='targets')

        x = bn_conv3d(input_tensor, filters=16, strides=(1, 1, 1), kernel_size=(3, 3, 3),
                      padding='same', name='initial_convolution', is_training=self.is_training)

        x = tf.layers.max_pooling3d(x, pool_size=(3, 3, 3), strides=(1, 2, 2),
                                    padding='same', name='maxpool3d')

        x = tf.layers.dropout(x, rate=self.dropout_rate, training=self.is_training)
        x = self.dense_block(x, filters=32, block_size=6, name='dense_block_1')
        x = self.transition_layer(x, filters=32, name='transition_layer_1')

        x = tf.layers.dropout(x, rate=self.dropout_rate, training=self.is_training)
        x = self.dense_block(x, filters=32, block_size=12, name='dense_block_2')
        x = self.transition_layer(x, filters=32, name='transition_layer_2')

        x = tf.layers.dropout(x, rate=self.dropout_rate, training=self.is_training)
        x = self.dense_block(x, filters=32, block_size=32, name='dense_block_3')
        x = self.transition_layer(x, filters=32, name='transition_layer_3')

        x = tf.layers.dropout(x, rate=self.dropout_rate, training=self.is_training)
        x = self.dense_block(x, filters=32, block_size=32, name='dense_block_4')
        x = self.transition_layer(x, filters=32, name='transition_layer_4')

        z = global_average_pool3d(x, name='global_average_pool3d')

        z = tf.layers.dense(z, units=self.num_targets, name='dense32')
        z = tf.nn.sigmoid(z)

        y_pred = tf.identity(z, name='predictions')

        self.store_to_attr('y', y_true)
        self.store_to_attr('x', input_tensor)
        self.store_to_attr('y_pred', y_pred)

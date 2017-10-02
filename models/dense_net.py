# pylint: disable=too-many-arguments
# pylint: disable=not-context-manager
""" Contains DenseNet model class. """

import tensorflow as tf
from .tf_model import TFModel, restore_nodes


class DenseNet(TFModel):
    """ This class implements 3D DenseNet architecture via tensorflow. """

    @staticmethod
    def max_pool3d(input_tensor, pool_size, strides, name, padding='same'):
        """ Apply max pooling 3D operation to input_tensor.

        Args:
        - input_tensor: tf.Variable, input_tensor;
        - pool_size: tuple(int, int, int), size of pooling kernel along 3 dimensions;
        - strides: tuple(int, int, int), size of strides along 3 dimensions;
        - name: str, name of layer that will be used as an argument of tf.variable_scope;

        Returns:
        - tf.Variable, output tensor;
        """
        with tf.variable_scope(name):
            out_tensor = tf.layers.max_pooling3d(input_tensor,
                                                 pool_size=pool_size,
                                                 strides=strides,
                                                 padding=padding,
                                                 name='maxpool3d')
        return out_tensor

    @staticmethod
    def average_pool3d(input_tensor, pool_size, strides, name, padding='same'):
        """ Apply average pooling 3D operation to input_tensor.

        Args:
        - input_tensor: tf.Variable, input tensor;
        - pool_size: tuple(int, int, int), size of pooling kernel along 3 dimensions;
        - strides: tuple(int, int, int), size of strides along 3 dimensions;
        - name: str, name of layer that will be used as an argument of tf.variable_scope;

        Returns:
        - tf.Variable, output tensor;
        """
        with tf.variable_scope(name):
            out_tensor = tf.layers.average_pooling3d(input_tensor,
                                                     pool_size=pool_size,
                                                     strides=strides,
                                                     padding='same',
                                                     name='average_pool3d')
        return out_tensor

    @staticmethod
    def global_average_pool3d(input_tensor, name):
        """ Apply global average pooling 3D operation to input tensor.

        Args:
        - input_tensor: tf.Variable, input tensor;
        - name: str, name of layer that will be used as an argument of tf.variable_scope;

        Returns:
        - tf.Variable, output tensor;
        """
        with tf.variable_scope(name):
            output_layer = tf.reduce_mean(input_tensor, axis=(1, 2, 3))
        return output_layer

    def dropout(self, input_tensor, rate=0.3):
        """ Add dropout layer with given dropout rate to the input tensor.

        Args:
        - input_tensor: tf.Variable, input tensor;
        - rate: float, dropout rate;

        Returns:
        - tf.Variable, output tensor;
        """
        return tf.layers.dropout(input_tensor, rate=rate, training=self.learning_phase)

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

        Args:
        - input_tensor: tf.Variable, input_tensor;
        - filters: int, number of filters in output tensor of each convolution layer;
        - block_size: int, how many time repeat conv3d{1x1x1} => conv3d{3x3x3}
        sequentially;
        - name: str, name of the layer that will be used as an argument of tf.variable_scope;

        Returns:
        - tf.Variable, output_tensor;
        """
        with tf.variable_scope(name):
            previous_input = tf.identity(input_tensor)
            for i in range(block_size):
                subblock_name = 'sub_block_' + str(i)
                x = self.bn_conv3d(previous_input, filters=filters,
                                   kernel_size=(1, 1, 1),
                                   strides=(1, 1, 1),
                                   name=subblock_name + '_conv3d_1_1',
                                   padding='same')

                x = self.bn_conv3d(x, filters=filters,
                                   kernel_size=(3, 3, 3),
                                   strides=(1, 1, 1),
                                   name=subblock_name + '_conv3d_3_3',
                                   padding='same')

                previous_input = tf.concat([previous_input, x], axis=-1)

        return previous_input

    def transition_layer(self, input_tensor, filters, name):
        """ Transition layer which is used as a dimension reduction block in densenset model.

        Args:
        - input_tensor: tf.Variable, input tensor;
        - filters: int, number of filters in output tensor;
        - name: str, name of the layer that will be used as an argument of tf.variable_scope;

        Returns:
        - tf.Variable, output tensor;
        """
        with tf.variable_scope(name):
            output_tensor = self.bn_conv3d(input_tensor, filters=filters,
                                           kernel_size=(1, 1, 1),
                                           strides=(1, 1, 1),
                                           name='conv3d_1_1')

            output_tensor = self.average_pool3d(output_tensor,
                                                strides=(2, 2, 2),
                                                pool_size=(2, 2, 2),
                                                name='averagepool_2_2')
        return output_tensor

    @restore_nodes('input', 'y_true', 'y_pred')
    def build_model(self):
        """ Build densenet model implemented via tensorflow. """
        input_tensor = tf.placeholder(shape=(None, 32, 64, 64, 1),
                                      dtype=tf.float32, name='input')

        y_true = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y_true')

        x = self.conv3d(input_tensor, filters=16, strides=(1, 1, 1), kernel_size=(3, 3, 3),
                        padding='same', name='initial_convolution')

        x = self.max_pool3d(x, pool_size=(3, 3, 3), strides=(1, 2, 2),
                            padding='same', name='initial_pooling')

        x = tf.layers.dropout(x, rate=0.35, training=self.learning_phase)
        x = self.dense_block(x, filters=32, block_size=6, name='dense_block_1')
        x = self.transition_layer(x, filters=32, name='transition_layer_1')

        x = tf.layers.dropout(x, rate=0.35, training=self.learning_phase)
        x = self.dense_block(x, filters=32, block_size=12, name='dense_block_2')
        x = self.transition_layer(x, filters=32, name='transition_layer_2')

        x = tf.layers.dropout(x, rate=0.35, training=self.learning_phase)
        x = self.dense_block(x, filters=32, block_size=32, name='dense_block_3')
        x = self.transition_layer(x, filters=32, name='transition_layer_3')

        x = tf.layers.dropout(x, rate=0.35, training=self.learning_phase)
        x = self.dense_block(x, filters=32, block_size=32, name='dense_block_4')
        x = self.transition_layer(x, filters=32, name='transition_layer_4')

        z = self.global_average_pool3d(x, name='global_average_pool3d')

        z = tf.layers.dense(z, units=1, name='dense32=>1')
        z = tf.nn.sigmoid(z)

        y_pred = tf.identity(z, name='y_pred')

        return input_tensor, y_true, y_pred

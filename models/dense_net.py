""" Contains DenseNet model class. """

import tensorflow as tf

from .tf_model import TFModel
from .tf_model import model_scope


def log_loss(y_true, y_pred, epsilon=10e-7):
    """ Log loss implemented in tensorflow.

    Args:
    - y_true: tf.Variable, true labels;
    - y_pred: tf.Variable, predicted logits;
    - epsilon: float, epsilon for avoiding computing log(0);

    Returns:
    - tf.Variable, log_loss;
    """
    return - tf.reduce_mean(y_true * tf.log(y_pred + epsilon)
                            + (1 - y_true) * tf.log(1 - y_pred + epsilon))


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
        with tf.variable_scope(name):  # pylint disable=not-context-manager
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
        with tf.variable_scope(name):  # pylint disable=not-context-manager
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
        with tf.variable_scope(name):  # pylint disable=not-context-manager
            output_layer = tf.reduce_mean(input_tensor, axis=(1, 2, 3))
        return output_layer

    @staticmethod
    def conv3d(input_tensor, filters, kernel_size, name,
               strides=(1, 1, 1), padding='same', activation=tf.nn.relu, use_bias=True):
        """ Apply 3D convolution operation to input tensor.

        Args:
        - input_tensor: tf.Variable, input tensor;
        - filters: int, number of filters in the ouput tensor;
        - kernel_size: tuple(int, int, int), size of kernel
          of 3D convolution operation along 3 dimensions;
        - name: str, name of the layer that will be used as an argument of tf.variable_scope;
        - strides: tuple(int, int, int), size of strides along 3 dimensions;
        - padding: str, padding mode, can be 'same' or 'valid';
        - activation: tensorflow activation function that will be applied to
        output tensor;
        - use_bias: bool, whether use bias or not;

        Returns:
        - tf.Variable, output tensor;
        """
        with tf.variable_scope(name):  # pylint disable=not-context-manager
            output_tensor = tf.layers.conv3d(input_tensor, filters=filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             use_bias=use_bias,
                                             name='conv3d', padding=padding)

            output_tensor = activation(output_tensor)
        return output_tensor

    def bn_conv3d(self, input_tensor, filters, kernel_size, name,
                  strides=(1, 1, 1), padding='same', activation=tf.nn.relu, use_bias=False):
        """ Apply 3D convolution operation with batch normalization to input tensor.

        Args:
        - input_tensor: tf.Variable, input tensor;
        - filters: int, number of filters in the ouput tensor;
        - kernel_size: tuple(int, int, int), size of kernel
          of 3D convolution operation along 3 dimensions;
        - name: str, name of the layer that will be used as an argument of tf.variable_scope;
        - strides: tuple(int, int, int), size of strides along 3 dimensions;
        - padding: str, padding mode, can be 'same' or 'valid';
        - activation: tensorflow activation function that will be applied to
        output tensor;
        - use_bias: bool, whether use bias or not;

        Returns:
        - tf.Variable, output tensor;
        """
        with tf.variable_scope(name):  # pylint disable=not-context-manager
            output_tensor = tf.layers.conv3d(input_tensor, filters=filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             use_bias=use_bias,
                                             name='conv3d', padding=padding)

            output_tensor = tf.layers.batch_normalization(output_tensor, axis=-1,
                                                          training=self.learning_phase)
            output_tensor = activation(output_tensor)
        return output_tensor

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
        concat([input, output_1, ..., output_n]) => conv3D{1x1x1}[1:1:1](filters) =>
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
        with tf.variable_scope(name):  # pylint disable=not-context-manager
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
        with tf.variable_scope(name):  # pylint disable=not-context-manager
            output_tensor = self.bn_conv3d(input_tensor, filters=filters,
                                           kernel_size=(1, 1, 1),
                                           strides=(1, 1, 1),
                                           name='conv3d_1_1')

            output_tensor = self.average_pool3d(output_tensor,
                                                strides=(2, 2, 2),
                                                pool_size=(2, 2, 2),
                                                name='averagepool_2_2')
        return output_tensor

    @model_scope
    def build_model(self):
        """ Build densenet model implemented via tensorflow. """
        input_tensor = tf.placeholder(shape=(None, 32, 64, 64, 1),
                                      dtype=tf.float32, name='input')
        y_true = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y_true')

        x = self.conv3d(input_tensor, filters=16, strides=(1, 1, 1), kernel_size=(5, 5, 5),
                        padding='same', name='initial_convolution')

        x = self.max_pool3d(x, pool_size=(3, 3, 3), strides=(1, 2, 2),
                            padding='same', name='initial_pooling')

        x = self.dense_block(x, filters=32, block_size=6, name='dense_block_1')
        x = self.transition_layer(x, filters=32, name='transition_layer_1')

        x = self.dense_block(x, filters=32, block_size=12, name='dense_block_2')
        x = self.transition_layer(x, filters=32, name='transition_layer_2')

        x = self.dense_block(x, filters=32, block_size=32, name='dense_block_3')
        x = self.transition_layer(x, filters=32, name='transition_layer_3')


        x = self.dense_block(x, filters=32, block_size=32, name='dense_block_4')
        x = self.transition_layer(x, filters=32, name='transition_layer_4')

        y_pred = self.global_average_pool3d(x, name='global_average_pool3d')

        y_pred = tf.layers.dense(y_pred, units=1, name='dense32_1')
        y_pred = tf.nn.sigmoid(y_pred)
        y_pred = tf.identity(y_pred, name='y_pred')

        self.input = input_tensor
        self.y_true = y_true
        self.y_pred = y_pred
        self.loss = log_loss(self.y_true, self.y_pred)

        self.add_to_collection((self.input, self.y_true, self.y_pred, self.loss))
        return self

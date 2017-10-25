# pylint: disable=too-many-arguments
# pylint: disable=not-context-manager
# pylint: disable=duplicate-code
""" Contains TFResNet model class. """

import tensorflow as tf
from ..tf_model import TFModel
from ..layers import conv3d, bn_conv3d, global_average_pool3d


class TFResNet(TFModel):
    """ This class implements 3D ResNet architecture via tensorflow. """

    def __init__(self, name, num_targets=1, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.num_targets = num_targets

    def identity_block(self, input_tensor, kernel_size, filters, name):
        """ The identity block is the block that has no conv layer at shortcut. """
        filters1, filters2, filters3 = filters

        with tf.variable_scope(name):
            x = bn_conv3d(input_tensor, filters1, (1, 1, 1),
                          name='bn_conv_a', padding='same',
                          activation=tf.nn.relu,
                          is_training=self.is_training)

            x = bn_conv3d(x, filters2, kernel_size,
                          name='bn_conv_b', padding='same',
                          activation=tf.nn.relu,
                          is_training=self.is_training)

            x = bn_conv3d(x, filters3, (1, 1, 1),
                          name='bn_conv_c', padding='same',
                          activation=tf.nn.relu,
                          is_training=self.is_training)

            output_tensor = tf.add(x, input_tensor)
        return output_tensor

    def conv_block(self, input_tensor, kernel_size, filters, name, strides=(2, 2, 2)):
        """ Convolutional block that has a conv layer as shortcut.

        Args:
        - input_tensor: tf.Variable, input tensor;
        - kernel_size: tuple(int, int, int), size of kernel
        of 3D convolution along 3 dimension of the middle layer;
        - filetrs: tuple(int, int, int)
        """
        filters1, filters2, filters3 = filters

        with tf.variable_scope(name):
            x = bn_conv3d(input_tensor, filters1, (1, 1, 1),
                          name='bn_conv_a', padding='same',
                          activation=tf.nn.relu, strides=strides,
                          is_training=self.is_training)

            x = bn_conv3d(x, filters2, kernel_size,
                          name='bn_conv_b', padding='same',
                          activation=tf.nn.relu, is_training=self.is_training)

            x = bn_conv3d(x, filters3, (1, 1, 1),
                          name='bn_conv_c', padding='same',
                          activation=tf.nn.relu)

            shortcut = bn_conv3d(input_tensor, filters3, (1, 1, 1),
                                 strides=strides, name='bn_conv3d_shortcut',
                                 padding='same', activation=tf.identity,
                                 is_training=self.is_training)

            output_tensor = tf.add(x, shortcut)
            output_tensor = tf.nn.relu(output_tensor)
        return output_tensor

    def build_model(self):
        """ Build renset model implemented via tensorflow. """
        input_tensor = tf.placeholder(shape=(None, 32, 64, 64, 1), dtype=tf.float32, name='input')
        y_true = tf.placeholder(shape=(None, self.num_targets), dtype=tf.float32, name='y_true')

        x = bn_conv3d(input_tensor, filters=32, kernel_size=(7, 3, 3),
                      name='initial_conv', padding='same',
                      is_training=self.is_training)

        x = tf.layers.max_pooling3d(x, pool_size=(3, 3, 3), strides=(2, 2, 2),
                                    name='initial_maxpool')

        x = self.conv_block(x, (3, 3, 3), [16, 16, 32], name='conv_1A', strides=(1, 1, 1))
        x = self.identity_block(x, (3, 3, 3), [16, 16, 32], name='identity_1B')
        x = self.identity_block(x, (3, 3, 3), [16, 16, 32], name='identity_1C')
        x = tf.layers.dropout(x, rate=0.35, training=self.is_training)

        x = self.conv_block(x, (3, 3, 3), [24, 24, 64], name='conv_2A')
        x = self.identity_block(x, (3, 3, 3), [24, 24, 64], name='identity_2B')
        x = self.identity_block(x, (3, 3, 3), [24, 24, 64], name='identity_2C')
        x = self.identity_block(x, (3, 3, 3), [24, 24, 64], name='identity_2D')
        x = tf.layers.dropout(x, rate=0.35, training=self.is_training)

        x = self.conv_block(x, (3, 3, 3), [48, 48, 128], name='conv_3A')
        x = self.identity_block(x, (3, 3, 3), [48, 48, 128], name='identity_3B')
        x = self.identity_block(x, (3, 3, 3), [48, 48, 128], name='identity_3C')

        x = self.identity_block(x, (3, 3, 3), [48, 48, 128], name='identity_3D')
        x = self.identity_block(x, (3, 3, 3), [48, 48, 128], name='identity_3E')
        x = self.identity_block(x, (3, 3, 3), [48, 48, 128], name='identity_3F')
        x = tf.layers.dropout(x, rate=0.35, training=self.is_training)

        x = self.conv_block(x, (3, 3, 3), [64, 64, 196], name='conv_4A')
        x = self.identity_block(x, (3, 3, 3), [64, 64, 196], name='identity_4B')
        x = self.identity_block(x, (3, 3, 3), [64, 64, 196], name='identity_4C')

        z = tf.contrib.layers.flatten(x)

        z = tf.layers.dense(z, units=64, name='dense_64')
        z = tf.layers.batch_normalization(z, axis=-1, training=self.is_training)
        z = tf.nn.relu(z)

        z = tf.layers.dense(z, units=16, name='dense_16')
        z = tf.layers.batch_normalization(z, axis=-1, training=self.is_training)
        z = tf.nn.relu(z)

        z = tf.layers.dense(z, units=self.num_targets, name='dense_' + str(self.num_targets))
        z = tf.nn.sigmoid(z)

        y_pred = tf.identity(z, name='y_pred')

        return input_tensor, y_true, y_pred

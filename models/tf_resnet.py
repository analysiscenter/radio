import numpy as np
import tensorflow as tf

from .tf_model import TFModel
from .tf_model import model_scope


def log_loss(y_true, y_pred, epsilon=10e-7):
    return - tf.reduce_mean(y_true * tf.log(y_pred + epsilon)
                            + (1 - y_true) * tf.log(1 - y_pred + epsilon))


class TFResNet(TFModel):

    @staticmethod
    def conv3d(input_tensor, filters, kernel_size, name,
               strides=(1, 1, 1), padding='same', activation=tf.nn.relu, use_bias=True):
        """ 3D convolution layer. """
        with tf.variable_scope(name):
            output_tensor = tf.layers.conv3d(input_tensor, filters=filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             use_bias=use_bias,
                                             name='conv3d', padding=padding)

            output_tensor = activation(output_tensor)
        return output_tensor

    def bn_conv3d(self, input_tensor, filters, kernel_size, name,
                  strides=(1, 1, 1), padding='same', activation=tf.nn.relu, use_bias=False):
        """ 3D convolution layer with batch normalization. """
        with tf.variable_scope(name):
            output_tensor = tf.layers.conv3d(input_tensor, filters=filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             use_bias=use_bias,
                                             name='conv3d', padding=padding)

            output_tensor = tf.layers.batch_normalization(output_tensor, axis=-1,
                                                          training=self.learning_phase)
            output_tensor = activation(output_tensor)
        return output_tensor

    def identity_block(self, input_tensor, kernel_size, filters, name):
        """ The identity block is the block that has no conv layer at shortcut. """
        filters1, filters2, filters3 = filters

        with tf.variable_scope(name):
            x = self.bn_conv3d(input_tensor, filters1, (1, 1, 1),
                               name='bn_conv_a', padding='same',
                               activation=tf.nn.relu)

            x = self.bn_conv3d(x, filters2, kernel_size,
                               name='bn_conv_b', padding='same',
                               activation=tf.nn.relu)

            x = self.bn_conv3d(x, filters3, (1, 1, 1),
                               name='bn_conv_c', padding='same',
                               activation=tf.nn.relu)

            output_tensor = tf.add(x, input_tensor)
        return output_tensor

    def conv_block(self, input_tensor, kernel_size, filters, name, strides=(2, 2, 2)):
        """ A block that has a conv layer at shortcut. """
        filters1, filters2, filters3 = filters

        with tf.variable_scope(name):
            x = self.bn_conv3d(input_tensor, filters1, (1, 1, 1),
                               name='bn_conv_a', padding='same',
                               activation=tf.nn.relu, strides=strides)

            x = self.bn_conv3d(x, filters2, kernel_size,
                               name='bn_conv_b', padding='same',
                               activation=tf.nn.relu)

            x = self.bn_conv3d(x, filters3, (1, 1, 1),
                               name='bn_conv_c', padding='same',
                               activation=tf.nn.relu)

            shortcut = self.bn_conv3d(input_tensor, filters3, (1, 1, 1),
                                      strides=strides, name='bn_conv3d_shortcut',
                                      padding='same', activation=tf.identity)
            output_tensor = tf.add(x, shortcut)
            output_tensor = tf.nn.relu(output_tensor)
        return output_tensor

    @model_scope
    def build_model(self):
        """ Build renset model implemented in tensorflow. """
        input_tensor = tf.placeholder(shape=(None, 32, 64, 64, 1), dtype=tf.float32, name='input')
        y_true = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y_true')

        x = self.bn_conv3d(input_tensor, filters=32,kernel_size=(7, 3, 3),
                        name='initial_conv', padding='same')

        x = tf.layers.max_pooling3d(x, pool_size=(3, 3, 3), strides=(2, 2, 2),
                                   name='initial_maxpool')

        x = self.conv_block(x, (3, 3, 3), [16, 16, 32], name='conv_1A', strides=(1, 1, 1))
        x = self.identity_block(x, (3, 3, 3), [16, 16, 32], name='identity_1B')
        x = self.identity_block(x, (3, 3, 3), [16, 16, 32], name='identity_1C')
        x = tf.layers.dropout(x, rate=0.35, training=self.learning_phase)

        x = self.conv_block(x, (3, 3, 3), [24, 24, 64], name='conv_2A')
        x = self.identity_block(x, (3, 3, 3), [24, 24, 64], name='identity_2B')
        x = self.identity_block(x, (3, 3, 3), [24, 24, 64], name='identity_2C')
        x = self.identity_block(x, (3, 3, 3), [24, 24, 64], name='identity_2D')
        x = tf.layers.dropout(x, rate=0.35, training=self.learning_phase)

        x = self.conv_block(x, (3, 3, 3), [48, 48, 128], name='conv_3A')
        x = self.identity_block(x, (3, 3, 3), [48, 48, 128], name='identity_3B')
        x = self.identity_block(x, (3, 3, 3), [48, 48, 128], name='identity_3C')
        # x = tf.layers.dropout(x, rate=0.35, training=self.learning_phase)
        x = self.identity_block(x, (3, 3, 3), [48, 48, 128], name='identity_3D')
        x = self.identity_block(x, (3, 3, 3), [48, 48, 128], name='identity_3E')
        x = self.identity_block(x, (3, 3, 3), [48, 48, 128], name='identity_3F')
        x = tf.layers.dropout(x, training=self.learning_phase)

        x = self.conv_block(x, (3, 3, 3), [64, 64, 196], name='conv_4A')
        x = self.identity_block(x, (3, 3, 3), [64, 64, 196], name='identity_4B')
        x = self.identity_block(x, (3, 3, 3), [64, 64, 196], name='identity_4C')
        # x = tf.layers.dropout(x, rate=0.35, training=self.learning_phase)

        z = tf.contrib.layers.flatten(x)

        z = tf.layers.dense(z, units=64, name='dense_128')
        z = tf.layers.batch_normalization(z, axis=-1, training=self.learning_phase)
        z = tf.nn.relu(z)
        # z = tf.layers.dropout(z, rate=0.35, training=self.training_phase)

        z = tf.layers.dense(z, units=16, name='dense_32')
        z = tf.layers.batch_normalization(z, axis=-1, training=self.learning_phase)
        z = tf.nn.relu(z)

        z = tf.layers.dense(z, units=1, name='dense_1')
        z = tf.nn.sigmoid(z)

        self.input = input_tensor
        self.y_true = y_true
        self.y_pred = z
        self.loss = log_loss(self.y_true, self.y_pred)

        self.add_to_collection((self.input, self.y_true, self.y_pred, self.loss))
        return self

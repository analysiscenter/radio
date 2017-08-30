""" Contains DenseNet model class. """

import numpy as np
import tensorflow as tf
from collections import namedtuple

from .tf_model import TFModel
from .tf_model import model_scope


def get_shape(x):
    """ Get shape of tensor passed as argument. """
    return x.get_shape().as_list()


def log_loss(y_true, y_pred, epsilon=10e-7):
    """ Log loss implemented in tensorflow. """
    return - tf.reduce_mean(y_true * tf.log(y_pred + epsilon)
                            + (1 - y_true) * tf.log(1 - y_pred + epsilon))


class DenseNet(TFModel):
    """ This class implements 3D DenseNet architecture via tensorflow. """

    @staticmethod
    def maxpool3d(input_tensor, pool_size, strides, name, padding='same'):
        """ Apply maxpooling3d operation to the input_tensor. """
        with tf.variable_scope(name):
            out_tensor = tf.layers.max_pooling3d(input_tensor,
                                                 pool_size=pool_size,
                                                 strides=strides,
                                                 padding=padding,
                                                 name='maxpool3d')
        return out_tensor

    @staticmethod
    def averagepool3d(input_tensor, pool_size, strides, name, padding='same'):
        """ Apply averagepool3d operation to the input_tensor. """
        with tf.variable_scope(name):
            out_tensor = tf.layers.average_pooling3d(input_tensor,
                                                     pool_size=pool_size,
                                                     strides=strides,
                                                     padding='same',
                                                     name='average_pool3d')
        return out_tensor

    @staticmethod
    def global_averagepool3d(input_tensor, name):
        """ Global average pooling 3d layer. """
        with tf.variable_scope(name):
            output_layer = tf.reduce_mean(input_tensor, axis=(1, 2, 3))
        return output_layer

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
        """ 3D convolution layer with batch normalization along last axis. """
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

    def dropout(self, input_tensor, rate=0.3):
        """ Add dropout layer with given dropout rate to the input tensor. """
        return tf.layers.dropout(input_tensor, rate=rate, training=self.learning_phase)

    def dense_block(self, input_tensor, filters, block_size, name):
        """ Dense block which is used as a build block of densenet model. """
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
        """ Transition layer which is used as a build block of densenet model. """
        with tf.variable_scope(name):
            output_tensor = self.bn_conv3d(input_tensor, filters=filters,
                                           kernel_size=(1, 1, 1),
                                           strides=(1, 1, 1),
                                           name='conv3d_1_1')

            output_tensor = self.averagepool3d(output_tensor,
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

        x = self.maxpool3d(x, pool_size=(3, 3, 3), strides=(1, 2, 2),
                           padding='same', name='initial_pooling')

        x = self.dense_block(x, filters=32, block_size=6, name='dense_block_1')
        x = self.transition_layer(x, filters=32, name='transition_layer_1')

        x = self.dense_block(x, filters=32, block_size=12, name='dense_block_2')
        x = self.transition_layer(x, filters=32, name='transition_layer_2')

        x = self.dense_block(x, filters=32, block_size=32, name='dense_block_3')
        x = self.transition_layer(x, filters=32, name='transition_layer_3')


        x = self.dense_block(x, filters=32, block_size=32, name='dense_block_4')
        x = self.transition_layer(x, filters=32, name='transition_layer_4')

        y_pred = self.global_averagepool3d(x, name='global_average_pool3d')

        y_pred = tf.layers.dense(y_pred, units=1, name='dense32_1')
        y_pred = tf.nn.sigmoid(y_pred)
        y_pred = tf.identity(y_pred, name='y_pred')

        self.input = input_tensor
        self.y_true = y_true
        self.y_pred = y_pred
        self.loss = log_loss(self.y_true, self.y_pred)

        self.add_to_collection((self.input, self.y_true, self.y_pred, self.loss))
        return self

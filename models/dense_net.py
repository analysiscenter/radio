import numpy as np
import tensorflow as tf

from .tf_model import TFModel
from .tf_model import get_activation
from .tf_model import model


class DenseNet(TFModel):

    def dense_block(self, input_tensor, filters, block_size, name):
        with tf.variable_scope(name):
            previous_input = tf.identity(input_tensor)
            for i in range(block_size):
                subblock_name = 'sub_block_' + str(i)
                x = self.bn_conv3d(previous_input, filters=filters,
                                   kernel_size=(1, 1, 1),
                                   strides=(1, 1, 1),
                                   name=subblock_name + '_conv3d_1_1',
                                   padding='same',
                                   activation='relu')

                x = self.bn_conv3d(x, filters=filters,
                                   kernel_size=(3, 3, 3),
                                   strides=(1, 1, 1),
                                   name=subblock_name + '_conv3d_3_3',
                                   padding='same',
                                   activation='relu')

                previous_input = tf.concat([previous_input, x], axis=-1)

        return previous_input

    def transition_layer(self, input_tensor, filters, name):
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

    @model
    def build_densenet(self):
        input_tensor = tf.placeholder(shape=(None, 32, 64, 64, 1), dtype=tf.float32, name='x')
        y_true = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y_true')

        x = self.conv3d(input_tensor, filters=64, strides=(1, 1, 1), kernel_size=(5, 5, 5),
                        padding='same', name='initial_convolution')

        x = self.dense_block(x, filters=32, block_size=6, name='dense_block_1')
        x = self.transition_layer(x, filters=32, name='transition_layer_1')
        print(get_shape(x))

        x = self.dense_block(x, filters=32, block_size=12, name='dense_block_2')
        x = self.transition_layer(x, filters=32, name='transition_layer_2')
        print(get_shape(x))

        x = self.dense_block(x, filters=32, block_size=32, name='dense_block_3')
        x = self.transition_layer(x, filters=32, name='transition_layer_3')
        print(get_shape(x))

        x = self.dense_block(x, filters=32, block_size=32, name='dense_block_4')
        x = self.transition_layer(x, filters=32, name='transition_layer_4')
        print(get_shape(x))

        y_pred = self.global_averagepool3d(x, name='global_average_pool3d')

        y_pred = self.dense(y_pred, units=1, name='dense32_1', activation='linear')
        y_pred = tf.identity(y_pred, name='y_pred')
        return input_tensor, y_true, y_pred

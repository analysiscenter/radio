import os
import sys
from functools import wraps
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from .base_model import BaseModel


def log_loss(y_true, y_pred, epsilon=10e-7):
    return -tf.reduce_mean(y_true * tf.log(y_pred + epsilon) + (1 - y_true) * tf.log(1 - y_pred + epsilon))


def mse(y_true, y_pred):
    return 0.5 * tf.reduce_mean((y_pred - y_true) ** 2)


def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred - y_true))


ACTIVATIONS = {'relu': tf.nn.relu,
               'selu': tf.identity,
               'linear': tf.identity,
               'sigmoid': tf.nn.sigmoid,
               'softmax': tf.nn.softmax,
               'tanh': tf.nn.tanh}


OPTIMIZERS = {'adam': tf.train.AdamOptimizer(),
              'rmsprop': tf.train.RMSPropOptimizer(0.005),
              'momentum': tf.train.MomentumOptimizer(0.005, 0.1)}


LOSSES = {'log_loss': log_loss, 'mse': mse}


def get_activation(activation_str):
    """ Get tf activation function by its name. """
    if activation_str is None:
        return lambda x: x
    return ACTIVATIONS[activation_str]


def get_shape(input_tensor):
    """ Return full shape of the input tensor represented by tuple of ints.

    Args:
    - input_tensor: tf.Variable, input_tensor.

    Return:
    - shape of input_tensor, tuple(int);
    """
    return input_tensor.get_shape().as_list()


def get_optimizer(optimizer_name):
    """ Get optimizer by name. """
    if isinstance(optimizer_name, str):
        return OPTIMIZERS[optimizer_name]
    else:
        return optimizer_name


def get_loss(loss_name):
    """ Get loss by name. """
    if isinstance(loss_name, str):
        return LOSSES[loss_name]
    else:
        return loss_name


def model(method):
    """ Wrap method of TFModel to apply all ops in context of model's graph and scope. """
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.graph.as_default():
            with tf.variable_scope(self.name):
                result = method(self, *args, **kwargs)
        return result
    return wrapped


class TFModel(BaseModel):

    def __new__(cls, name, *args, **kwargs):
        """ Add necessary attributes to object of tensorflow model. """
        instance = super(TFModel, cls).__new__(cls)

        instance.name = name

        graph = tf.Graph()
        instance._graph = graph
        with graph.as_default():
            with tf.variable_scope(name):
                instance._phase_placeholder = tf.placeholder(tf.bool)
                instance._sess = None

                instance.phase = True
                instance.train_op = None
                instance.loss = None
                instance.x = None
                instance.y_pred = None
                instance.y_true = None

        return instance

    @property
    def graph(self):
        """ Get tensorflow graph object. """
        return self._graph

    @property
    def sess(self):
        """ Get tensorflow session. """
        return self._sess

    @property
    def learning_phase(self):
        """ Get learning phase represented by tf.placeholder(tf.bool). """
        return self._phase_placeholder

    def get_number_of_trainable_vars(self):
        with self.graph.as_default():
            arr = np.asarray([np.prod(get_shape(v)) for v in tf.trainable_variables()])
        return np.sum(arr)

    @staticmethod
    def flatten(input_tensor):
        """ Flatten input tensor.

        Args:
        - input_tensor: tf.Variable, input tensor;

        Return:
        - tf.Variable, flattened input tensor;
        """
        return tf.contrib.layers.flatten(input_tensor)

    @staticmethod
    def num_channels(input_tensor):
        """ Return channels dimension of the input tensor.

        Args:
        - input_tensor: tf.Variable, input tensor.

        Return:
        - number of channels, int;
        """
        return input_tensor.get_shape().as_list()[-1]

    @staticmethod
    def batch_size(input_tensor):
        """ Return batch size of the input tensor represented by first dimension.

        Args:
        - input_tensor: tf.Variable, input_tensor.

        Return:
        - number of items in the batch, int;
        """
        return input_tensor.get_shape().as_list()[0]

    @staticmethod
    def get_shape(input_tensor):
        """ Return full shape of the input tensor represented by tuple of ints.

        Args:
        - input_tensor: tf.Variable, input_tensor.

        Return:
        - shape of input_tensor, tuple(int);
        """
        return input_tensor.get_shape().as_list()

    @staticmethod
    def identity(input_tensor, name):
        """ Create an alias with given name for input_tensor.

        Args:
        - input_tensor: tf.Variable, input tensor;
        - name: str, name of alias;

        Return:
        - alias of input tensor, tf.Variable;
        """
        return tf.identity(input_tensor, name=name)

    def dense(self, input_tensor, units, name, activation='linear'):
        """ Wrap input tensor with dense layer.

        Args:
        - input_tensor: tf.Variable, input tensor;
        - units: int, number of units in the output tensor;
        - name: str, name of the dense layer's scope;
        - activation: str, activation to put after convolution;

        Return:
        - output tensor, tf.Variable;
        """
        with tf.variable_scope('Dense', name):
            init_w = tf.truncated_normal(shape=(self.num_channels(input_tensor),
                                                units), dtype=tf.float32)
            w = tf.Variable(init_w)
            b = tf.Variable(tf.random_uniform(shape=(units, ), dtype=tf.float32))

            output_tensor = tf.matmul(input_tensor, w) + b
            output_tensor = get_activation(activation)(output_tensor)
        return output_tensor

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
        with tf.variable_scope(name):
            output_layer = tf.reduce_mean(input_tensor, axis=(1, 2, 3))
        return output_layer

    @staticmethod
    def conv3d(input_tensor, filters, kernel_size, name,
               strides=(1, 1, 1), padding='same', activation=None, use_bias=True):
        activation_fn = get_activation(activation)
        with tf.variable_scope(name):
            output_tensor = tf.layers.conv3d(input_tensor, filters=filters,
                                          kernel_size=kernel_size,
                                          strides=strides,
                                          use_bias=use_bias,
                                          name='conv3d', padding=padding)

            output_tensor = activation_fn(output_tensor)
        return output_tensor

    def bn_conv3d(self, input_tensor, filters, kernel_size, name,
                  strides=(1, 1, 1), padding='same', activation=None, use_bias=False):
        activation_fn = get_activation(activation)
        with tf.variable_scope(name):
            output_tensor = tf.layers.conv3d(input_tensor, filters=filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             use_bias=use_bias,
                                             name='conv3d', padding=padding)

            output_tensor = tf.layers.batch_normalization(output_tensor, axis=-1,
                                                          training=self.learning_phase)
            output_tensor = activation_fn(output_tensor)
        return output_tensor

    def dropout(self, input_tensor, rate=0.3):
        """ Add dropout layer with given dropout rate to the input tensor. """
        return tf.layers.dropout(input_tensor, rate=rate, training=self.learning_phase)

    def train_on_batch(self, x, y_true, return_loss=False, callbacks=None, **kwargs):
        self.phase = True
        feed_dict = {self.x: x, self.y_true: y_true, self.learning_phase: self.phase}
        _, loss, y_pred = self.sess.run([self.train_op, self.loss, self.y_pred], feed_dict=feed_dict)
        if return_loss:
            return y_pred, loss
        else:
            return y_pred

    def test_on_batch(self, x, y_true, return_loss=False, callbacks=None, **kwargs):
        self.phase = False
        feed_dict = {self.x: x, self.learning_phase: self.phase}
        _, loss, y_pred = self.sess.run([self.train_op, self.loss, self.y_pred], feed_dict=feed_dict)
        if return_loss:
            return y_pred, loss
        else:
            return y_pred

    def predict_on_batch(self, x, **kwargs):
        self.phase = False
        feed_dict = {self.x: x, self.learning_phase: self.phase}
        _, y_pred = self.sess.run([self.train_op, self.y_pred], feed_dict=feed_dict)
        return y_pred

    def save(self, dir_path, *args, **kwargs):
        raise NotImplementedError

    def load(self, dir_path, *args, **kwargs):
        raise NotImplementedError

    @model
    def compile(self, optimizer, loss):
        _loss = get_loss(loss)
        _optimizer = get_optimizer(optimizer)

        model_dsc = self.build_model()
        y_true, y_pred = model_dsc.get('y_true'), model_dsc.get('y_pred')
        tf_loss = _loss(y_true, y_pred)

        self.x = model_dsc.get('x')
        self.loss = tf_loss
        self.y_pred = y_pred
        self.y_true = y_true

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = _optimizer.minimize(tf_loss)

        self._sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.train_op = train_op
        return self

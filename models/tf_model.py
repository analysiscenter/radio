# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=not-context-manager
""" Contains base class for all tensorflow models. """

import os
from functools import wraps
import json
import numpy as np
import tensorflow as tf

from .base_model import BaseModel


DECAY_DICT = {'exp': tf.train.exponential_decay, 'inverse_time': tf.train.inverse_time_decay}

class TFModel(BaseModel):
    """ Base class for all tensorflow models. """

    def __init__(self, name, *args, **kwargs):
        """ Initialize tensorflow model.

        1) Add self.graph = tf.Graph();
        2) Add self.name = name which will be used as root variable scope of model;
        3) Add self.global_step = 0;
        4) Add self.tensor_names = {};
        5) Add self.learning_phase = tf.placeholder(tf.bool);
        6) Initalize self.sess, self.train_step, self.y_pred,
        self.y_true, self.loss, self.input with None;
        """
        super().__init__(name, *args, **kwargs)
        self.name = name
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.input = None
            self.y_true = None
            self.y_pred = None

            self.sess = None
            self.loss = None
            self.train_step = None

            self.learning_rate = None
            self.restore_keys = {'vars': [], 'ops': []}
            self.learning_phase = tf.placeholder(tf.bool, name='learning_phase')
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            self.add_restore_var(self.learning_phase)
            self.add_restore_var(self.global_step)

    def set_decay(self, **kwargs):
        """ Set decay of learning rate. """
        with self.graph.as_default():
            decay_mode = 'exp'
            if 'decay_mode' in kwargs:
                decay_mode = kwargs.pop('decay_mode')
            decay_fn = DECAY_DICT[decay_mode]
            self.learning_rate = decay_fn(global_step=self.global_step, **kwargs)

    def add_restore_var(self, variable, alias=None):
        """ Enable tf.Tensor or tf.Variable to be restored after dump as an attribute.

        Args:
        - variable: tf.Tensor or tf.Variable to add to collection of variables
        that can be restored after dump;
        - alias: str, alias for tf.Variable or tf.Tensor which will be used as
        a name of attribute for accessing it from outside of TFModel instance;
        """
        if not isinstance(variable, (tf.Tensor, tf.Variable)):
            raise ValueError("Argument 'variable' must be an instance of class tf.Tensor")

        if alias is None:
            _alias = variable.name.split('/')[-1].split(':')[0]
        else:
            _alias = alias

        self.restore_keys['vars'].append(_alias)
        with self.graph.as_default():
            restore_vars_collection = tf.get_collection_ref('restore_vars')
            restore_vars_collection.append(variable)

    def add_restore_op(self, operation, alias=None):
        """ Enable tf.Operation to be restored after dump as an attribute.

        Args:
        - operation: tf.Operation, operation to add to collection of operations
        that can be restored after dump;
        - alias: str, alias for tf.Operation which will be used as a name of
        attribute for accessing this operation;
        """
        if not isinstance(operation, tf.Operation):
            raise ValueError("Argument 'operation' must be an instance of class tf.Operation")

        if alias is None:
            _alias = operation.name.split('/')[-1].split(':')[0]
        else:
            _alias = alias

        self.restore_keys['vars'].append(_alias)
        with self.graph.as_default():
            restore_ops_collection = tf.get_collection_ref('restore_ops')
            restore_ops_collection.append(operation)

    def build_model(self, *args, **kwargs):
        """ Build tensorflow model.

        This method must be implemented in ancestor class:
        inside it tensorflow model must be build,
        self.y_true, self.input, self.y_pred, self.loss attributes
        must be set as tensorflow tensors;

        NOTE: this method will be automatically called when compile method
        is called by user;
        """
        raise NotImplementedError()

    def get_number_of_trainable_vars(self):
        """ Get number of trainable variable in graph associated with current model. """
        with self.graph.as_default():
            arr = np.asarray([np.prod(self.get_shape(v)) for v in tf.trainable_variables()])
        return np.sum(arr)

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

    def train_on_batch(self, x, y_true, **kwargs):
        """ Train tensorflow model on batch data. """
        with self.graph.as_default():
            feed_dict = {self.input: x, self.y_true: y_true, self.learning_phase: True}
            _ = self.sess.run(self.train_step, feed_dict=feed_dict)

    def predict_on_batch(self, x, **kwargs):
        """ Get prediction of tensorflow model on batch data. """
        with self.graph.as_default():
            feed_dict = {self.input: x, self.learning_phase: False}
            y_pred = self.sess.run(self.y_pred, feed_dict=feed_dict)
        return y_pred

    def save(self, dir_path, *args, **kwargs):
        """ Save tensorflow model. """
        with self.graph.as_default():
            saver = tf.train.Saver()
            path = os.path.join(dir_path, self.name)
            saver.save(self.sess, path, global_step=self.global_step)
            with open(os.path.join(dir_path, 'restore_keys.json'), 'w') as f:
                json.dump(self.restore_keys, f)
            self.saver = saver
        return self

    def load(self, dir_path, graph_path, checkpoint=None, *args, **kwargs):
        """ Load tensorflow model. """
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(graph_path)

        if checkpoint is None:
            checkpoint_path = tf.train.latest_checkpoint(dir_path)
        else:
            checkpoint_path = os.path.join(dir_path, checkpoint)
        saver.restore(self.sess, checkpoint_path)
        self.graph = self.sess.graph

        with open(os.path.join(dir_path, 'restore_keys.json'), 'r') as json_file:
            self.restore_keys = json.load(json_file)

        with self.graph.as_default():
            for alias, var in zip(self.restore_keys['vars'], tf.get_collection('restore_vars')):
                setattr(self, alias, var)

            for alias, op in zip(self.restore_keys['ops'], tf.get_collection('restore_ops')):
                setattr(self, alias, op)

        return self

    def compile(self, optimizer, loss, *args, **kwargs):
        """ Compile tensorflow model. """
        with self.graph.as_default():

            for var in self.build_model(**kwargs):
                self.add_restore_var(var)

            self.loss = loss(self.y_true, self.y_pred)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.train_step = optimizer.minimize(self.loss, global_step=self.global_step)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

        self.add_restore_var(self.loss, alias='loss')
        self.add_restore_op(self.train_step, alias='train_step')

    def conv3d(self, input_tensor, filters, kernel_size, name,
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

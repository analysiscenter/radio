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


def model_scope(method):
    """ Wrap method of TFModel to apply all ops in context of model's graph and scope. """
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        """ Wrapped method. """
        with self.graph.as_default():
            with tf.variable_scope(self.name):
                result = method(self, *args, **kwargs)
        return result
    return wrapped


class TFModel(BaseModel):
    """ Base class for all tensorflow models. """

    def __init__(self, name, *args, **kwargs):
        """ Initialize tensorflow model.

        1) Add self.graph = tf.Graph();
        2) Add self.name = name which will be used as root variable scope of model;
        3) Add self.global_step = 0;
        4) Add self.tensor_names = {};
        5) Add self.learning_phase = tf.placeholder(tf.bool);
        6) Initalize self.sess, self.train_op, self.y_pred,
        self.y_true, self.loss, self.input with None;
        """
        super().__init__(name, *args, **kwargs)
        self.name = name
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(self.name):
                self.tensor_names = {}
                self.global_step = 0

                self.learning_phase = tf.placeholder(tf.bool)
                self.add_to_collection(self.learning_phase, 'learning_phase')

                self.sess = None
                self.train_op = None
                self.loss = None
                self.input = None
                self.y_pred = None
                self.y_true = None


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

    def add_to_collection(self, tensor, alias=None):
        """ Add tensor to inner collection. """
        tensor_list = tensor if isinstance(tensor, (list, tuple)) else [tensor]
        if alias is None:
            alias_list = [t.name.split('/')[-1].split(':')[0] for t in tensor_list]
        elif isinstance(alias, str):
            alias_list = [alias]
        self.tensor_names.update({a: t.name for t, a in zip(tensor_list, alias_list)})

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
        feed_dict = {self.input: x, self.y_true: y_true, self.learning_phase: True}
        _ = self.sess.run(self.train_op, feed_dict=feed_dict)

    def predict_on_batch(self, x, **kwargs):
        """ Get prediction of tensorflow model on batch data. """
        feed_dict = {self.input: x, self.learning_phase: False}
        y_pred = self.sess.run(self.y_pred, feed_dict=feed_dict)
        return y_pred

    @model_scope
    def save(self, dir_path, *args, **kwargs):
        """ Save tensorflow model. """
        saver = tf.train.Saver()
        path = os.path.join(dir_path, self.name)
        saver.save(self.sess, path, global_step=self.global_step)
        with open(os.path.join(dir_path, 'tensor_collection.json'), 'w') as f:
            json.dump(self.tensor_names, f)
        return self

    def load(self, graph_path, checkpoint_path, dir_path, *args, **kwargs):
        """ Load tensorflow model. """
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(graph_path)
        saver.restore(self.sess, checkpoint_path)
        self.graph = self.sess.graph

        with open(os.path.join(dir_path, 'tensor_collection.json'), 'r') as f:
            self.tensor_names = json.load(f)

        for alias, name in self.tensor_names.items():
            setattr(self, alias, self.graph.get_tensor_by_name(name))
        return self

    def compile(self, optimizer, *args, **kwargs):
        """ Compile tensorflow model. """
        self.build_model()

        with self.graph.as_default():
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(self.loss)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.train_op = train_op
        return self

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

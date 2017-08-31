# pylint: disable=too-many-arguments
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
        self.name = name
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(self.name):
                self.tensor_name = {}
                self.global_step = 0

                self.learning_phase = tf.placeholder(tf.bool)
                self.add_to_collection(self.learning_phase, 'learning_phase')

                self.sess = None
                self.train_op = None
                self.loss = None
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
            arr = np.asarray([np.prod(get_shape(v)) for v in tf.trainable_variables()])
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
        _, loss, y_pred = self.sess.run([self.train_op, self.loss, self.y_pred], feed_dict=feed_dict)

    def predict_on_batch(self, x, **kwargs):
        """ Get prediction of tensorflow model on batch data. """
        feed_dict = {self.input: x, self.learning_phase: False}
        y_pred = self.sess.run([self.y_pred], feed_dict=feed_dict)[0]
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

    @model_scope
    def compile(self, optimizer):
        """ Compile tensorflow model. """
        self.build_model()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.train_op = train_op
        return self

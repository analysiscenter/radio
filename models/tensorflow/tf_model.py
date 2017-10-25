# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=not-context-manager
""" Contains base class for all tensorflow models. """

import os
import functools
import json
import numpy as np
import tensorflow as tf

from ..dataset.models.tf import TFModel


DECAY_DICT = {'exp': tf.train.exponential_decay,
              'inverse_time': tf.train.inverse_time_decay}


def restore_nodes(*names):
    """ Decorator for making output tensors of TFModel method be restorable as attributes. """
    if any(not isinstance(arg, str) for arg in names):
        raise ValueError("Arguments of restore_nodes decorator must be strings "
                         + "that will be names of attributes to "
                         + "which output tensors from decorated method "
                         + "will assosiated with")

    def decorated(method):
        """ Decorator with captured names for ouput tensors. """
        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):
            """ Function that will be used insted original method. """
            out_tf_variables = method(self, *args, **kwargs)
            if not isinstance(out_tf_variables, (tuple, list)):
                out_tf_variables = (out_tf_variables, )
            for alias, variable in zip(names, out_tf_variables):
                self.add_restore_var(variable, alias)
                setattr(self, alias, variable)
            return (*out_tf_variables, )

        return wrapped
    return decorated


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
        self.graph_context = None
        with self.graph.as_default():

            self.input = None
            self.y_true = None
            self.y_pred = None

            self.sess = None
            self.loss = None
            self.train_step = None
            self.saver = None

            self.learning_rate = None
            self.restore_keys = {'vars': [], 'ops': []}
            self.learning_phase = tf.placeholder(tf.bool, name='learning_phase')
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            self.add_restore_var(self.learning_phase)
            self.add_restore_var(self.global_step)

    def __enter__(self):
        """ Use model context.

        This magic method enables using TFModels instances with with statements.
        Inside __enter__ just default tensorflow graph is changed to self.graph.

        Returns:
        - self, TFModel.
        """
        self.graph_context = self.graph.as_default()
        self.graph_context.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """ Exit model context.

        This magic method enables using TFModel instances with 'with' statements.
        Inside __exit__ just default tensorflow graph is restored.

        Args:
        - exception_type: type of exception that was raised inside context.
        - exception_value: exception instance.
        - exception_traceback: traceback of exception that was raised inside context.
        """
        return self.graph_context.__exit__(exception_type,  # pylint: disable=no-member
                                           exception_value,
                                           exception_traceback)

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

        self.restore_keys['ops'].append(_alias)
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
        - input_tensor: tf.Variable, input_tensor;

        Return:
        - shape of input_tensor, tuple(int);
        """
        return input_tensor.get_shape().as_list()

    def train_on_batch(self, x, y_true, **kwargs):
        """ Train tensorflow model on batch data.

        Args:
        - x: ndarray, numpy array that will be fed into input
        placeholder(self.input attribute by default);
        - y_true: ndarray, numpy array that will be fed into
        'true-value' placeholder(self.y_true attribute by default);

        Returns:
        - None;

        NOTE: **kwargs argument is added for compatibillity
        with BaseModel method.
        """
        with self.graph.as_default():
            feed_dict = {self.input: x, self.y_true: y_true, self.learning_phase: True}
            _ = self.sess.run(self.train_step, feed_dict=feed_dict)
        return None

    def predict_on_batch(self, x, **kwargs):
        """ Get prediction of tensorflow model on batch data.

        Args:
        - x: ndarray, numpy array that will be fed to input
        placeholder(self.input attribute by default);

        Returns:
        - y_pred: ndarray containing predictions of the model;

        NOTE: **kwargs argument is added for compatibillity
        with BaseModel method.
        """
        with self.graph.as_default():
            feed_dict = {self.input: x, self.learning_phase: False}
            y_pred = self.sess.run(self.y_pred, feed_dict=feed_dict)
        return y_pred

    def save(self, dir_path, *args, **kwargs):
        """ Save tensorflow model.

        Args:
        - dir_path: str, path to directory on hard drive that will be used
        for storing model's data;

        Returns:
        - self, TFModel instance;

        NOTE: *args and **kwargs are added for compatibillity with BaseModel class.

        Example:
        >>> tf_model = TFResNetModel('resnet50')
        >>> .............training model..............
        >>> tf_model.save('/path/to/resnet50/model/')
        """
        with self.graph.as_default():
            if self.saver is None:
                self.saver = tf.train.Saver()
            path = os.path.join(dir_path, self.name)
            self.saver.save(self.sess, path, global_step=self.global_step)
            with open(os.path.join(dir_path, 'restore_keys.json'), 'w') as f:
                json.dump(self.restore_keys, f)
        return self

    def load(self, dir_path, graph_path, checkpoint=None, *args, **kwargs):
        """ Load tensorflow model.

        Args:
        - dir_path: str, path to directory on hard drive where model
        was saved via save method;
        - graph_path: str, path to file with metagraph;
        - checkpoint: str or None, if str then must represent path
        to checkpoint file that will be loaded, if None then latest checkpoint
        from 'dir_path' directory will be loaded;

        Example:
        >>> tf_model = TFResNetModel('resnet50')
        >>> tf.model.load('/path/to/resnet50/model/', '/path/to/resnet50/model/resnet50-1982.meta')
        """
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
        """ Compile tensorflow model.

        First of all compile method enters in the context of graph that
        corresponds to the model. After that it calls build_model method.
        Finally, it assigns loss tensor(loss_tensor = loss(self.y_true, self.y_pred))
        to self.loss attribute and assigns result of optimizer.minimize(self.loss)
        operation to self.train_step attribute.

        NOTE: after assignment self.loss and self.train_step are added to
        collections of restorable tensors and operations.

        Args:
        - optimizer: optimizer from tf.train.*;
        - loss: tensorflow function that takes self.y_true as its first argument
        self.y_pred as its second argument and return real-value tensor as result.

        NOTE: it's strongly recommended to create optimizer and pass it into compile method
        inside model context(see Example)

        Example:
        >>> tf_model = TFResNetModel('resnet50')
        >>> with tf_model:
        ...     tf_model.compile(optimizer=tf.train.AdamOptimizer(0.005), loss=tf.losses.log_loss)

        that is equal to

        >>> tf_model = TFResNetModel('resnet50')
        >>> with tf.model.graph.as_default():
        ...     tf_model.compile(optimizer=tf.train.AdamOptimizer(0.005), loss=tf.losses.log_loss)

        but is shorter.
        """
        with self.graph.as_default():

            _ = self.build_model()
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

    def bn_dilated_conv3d(self, input_tensor, filters,
                          kernel_size, name, activation=tf.nn.relu,
                          dilation=(1, 1, 1), padding='same'):
        """ Apply 3D convolution operation with batch normalization to input tensor.

        Args:
        - input_tensor: tf.Variable, input tensor;
        - filters: int, number of filters in the ouput tensor;
        - kernel_size: tuple(int, int, int), size of kernel
          of 3D convolution operation along 3 dimensions;
        - name: str, name of the layer that will be used as an argument of tf.variable_scope;
        - dilation: tuple(int, int, int), size of dilation along 3 dimensions;
        - padding: str, padding mode, can be 'same' or 'valid';
        - activation: tensorflow activation function that will be applied to
        output tensor;

        Returns:
        - tf.Variable, output tensor;
        """

        in_filters = input_tensor.get_shape().as_list()[-1]
        with tf.variable_scope(name):
            w = tf.get_variable(name='W', shape=(*kernel_size, in_filters, filters),
                                initializer=tf.contrib.layers.xavier_initializer())

            output_tensor = tf.nn.convolution(input_tensor, w,
                                              padding=padding.upper(),
                                              strides=(1, 1, 1),
                                              dilation_rate=dilation)

            output_tensor = tf.layers.batch_normalization(output_tensor,
                                                          axis=4, training=self.learning_phase)

            output_tensor = activation(output_tensor)

        return output_tensor

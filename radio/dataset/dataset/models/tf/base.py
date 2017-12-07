# pylint: disable=undefined-variable
""" Contains base class for tensorflow models """

import os
import glob
import re
import json
import threading

import numpy as np
import tensorflow as tf

from ..base import BaseModel
from .layers import mip, conv_block
from .losses import dice


LOSSES = {
    'mse': tf.losses.mean_squared_error,
    'ce': tf.losses.softmax_cross_entropy,
    'crossentropy': tf.losses.softmax_cross_entropy,
    'absolutedifference': tf.losses.absolute_difference,
    'l1': tf.losses.absolute_difference,
    'cosine': tf.losses.cosine_distance,
    'cos': tf.losses.cosine_distance,
    'hinge': tf.losses.hinge_loss,
    'huber': tf.losses.huber_loss,
    'logloss': tf.losses.log_loss,
    'dice': dice
}

DECAYS = {
    'exp': tf.train.exponential_decay,
    'invtime': tf.train.inverse_time_decay,
    'naturalexp': tf.train.natural_exp_decay,
    'const': tf.train.piecewise_constant,
    'poly': tf.train.polynomial_decay
}



class TFModel(BaseModel):
    r""" Base class for all tensorflow models

    **Configuration**

    ``build`` and ``load`` are inherited from :class:`.BaseModel`.

    session : dict
        `Tensorflow session parameters <https://www.tensorflow.org/api_docs/python/tf/Session#__init__>`_.

    inputs : dict
        model inputs (see :meth:`._make_inputs`)

    loss - a loss function, might be one of:
        - short name (`'mse'`, `'ce'`, `'l1'`, `'cos'`, `'hinge'`, `'huber'`, `'logloss'`, `'dice'`)
        - a function name from `tf.losses <https://www.tensorflow.org/api_docs/python/tf/losses>`_
          (e.g. `'absolute_difference'` or `'sparse_softmax_cross_entropy'`)
        - callable

        Examples:

        - ``{'loss': 'mse'}``
        - ``{'loss': 'sigmoid_cross_entropy'}``
        - ``{'loss': tf.losses.huber_loss}``
        - ``{'loss': external_loss_fn}``

    decay - a learning rate decay algorithm might be defined in one of three formats:
        - name
        - tuple (name, args)
        - dict {'name': name, \**args}

        where name might be one of:

        - short name ('exp', 'invtime', 'naturalexp', 'const', 'poly')
        - a function name from `tf.train <https://www.tensorflow.org/api_docs/python/tf/train>`_
          (e.g. 'exponential_decay')
        - a callable

        Examples:

        - ``{'decay': 'exp'}``
        - ``{'decay': ('polynomial_decay', {'decay_steps':10000})}``
        - ``{'decay': {'name': tf.train.inverse_time_decay, 'decay_rate': .5}``

    optimizer - an optimizer might be defined in one of three formats:
            - name
            - tuple (name, args)
            - dict {'name': name, \**args}

            where name might be one of:

            - short name (e.g. 'Adam', 'Adagrad', any optimiser from
              `tf.train <https://www.tensorflow.org/api_docs/python/tf/train>`_ without a word `Optimiser`)
            - a function name from `tf.train <https://www.tensorflow.org/api_docs/python/tf/train>`_
              (e.g. 'FtlrOptimizer')
            - a callable

        Examples:

        - ``{'optimizer': 'Adam'}``
        - ``{'optimizer': ('Ftlr', {'learning_rate_power': 0})}``
        - ``{'optimizer': {'name': 'Adagrad', 'initial_accumulator_value': 0.01}``
        - ``{'optimizer': functools.partial(tf.train.MomentumOptimizer, momentum=0.95)}``
        - ``{'optimizer': some_optimizer_fn}``

    common : dict
        default parameters for all :func:`.conv_block`

    input_block : dict
        parameters for the input block, usually :func:`.conv_block` parameters.

        The only required parameter here is ``input_block/inputs`` which should contain a name or
        a list of names from ``inputs`` which tensors will be passed to ``input_block`` as ``inputs``.

        Examples:

        - ``{'input_block/inputs': 'images'}``
        - ``{'input_block': dict(inputs='features')}``
        - ``{'input_block': dict(inputs='images', layout='nac nac', filters=64, kernel_size=[7, 3], strides=[1, 2])}``

    body : dict
        parameters for the base network layers, usually :func:`.conv_block` parameters

    head : dict
        parameters for the head layers, usually :func:`.conv_block` parameters

    output : dict
        ops : tuple of str
            helper operations:

            - 'accuracy' - add 'accuracy' tensor with accuracy score for predictions
            - 'proba' - add 'predicted_proba' tensor with probabilties for predictions
            - 'labels' - add 'predicted_labels' tensor with best match labels for predictions


    **How to create your own model**

    #. Take a look at :func:`~.layers.conv_block` since it is widely used as a building block almost everywhere.

    #. Define model defaults (e.g. number of filters, batch normalization options, etc)
       by overriding :meth:`.TFModel.default_config`.
       Or skip it and hard code all the parameters in unpredictable places without the possibility to
       change them easily through model's config.

    #. Define build configuration (e.g. number of classes, etc)
       by overriding :meth:`~.TFModel.build_config`.

    #. Override :meth:`~.TFModel.input_block`, :meth:`~.TFModel.body` and :meth:`~.TFModel.head`, if needed.
       In many cases defaults and build config are just enough to build a network without additional code writing.

    Things worth mentioning:

    #. Input data and its parameters should be defined in configuration under ``inputs`` key.
       See :meth:`._make_inputs` for details.

    #. You might want to use a convenient multidimensional :func:`.conv_block`,
       as well as :func:`~.layers.global_average_pooling`, :func:`~.layers.mip`, or other predefined layers.
       Of course, you can use usual `tensorflow layers <https://www.tensorflow.org/api_docs/python/tf/layers>`_.

    #. If you make dropout, batch norm, etc by hand, you might use a predefined ``self.is_training`` tensor.

    #. For decay and training control there is a predefined ``self.global_step`` tensor.

    #. In many cases there is no need to write a loss function, learning rate decay and optimizer
       as they might be defined through config.

    #. For a configured loss one of the inputs should have a name ``targets`` and
       one of the tensors in your model should have a name ``predictions``.
       They will be used in a loss function.

    #. If you have defined your own loss function, call `tf.losses.add_loss(...)
       <https://www.tensorflow.org/api_docs/python/tf/losses/add_loss>`_.

    #. If you need to use your own optimizer, assign ``self.train_step`` to the train step operation.
       Don't forget about `UPDATE_OPS control dependency
       <https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization>`_.
    """

    def __init__(self, *args, **kwargs):
        self.session = kwargs.get('session', None)
        self.graph = tf.Graph() if self.session is None else self.session.graph
        self._graph_context = None
        self.is_training = None
        self.global_step = None
        self.loss = None
        self.train_step = None
        self._train_lock = threading.Lock()
        self._attrs = []
        self._to_classes = {}
        self._inputs = {}
        self.inputs = None

        super().__init__(*args, **kwargs)


    def __enter__(self):
        """ Enter the model graph context """
        self._graph_context = self.graph.as_default()
        return self._graph_context.__enter__()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """ Exit the model graph context """
        return self._graph_context.__exit__(exception_type, exception_value, exception_traceback)

    def build(self, *args, **kwargs):
        """ Build the model

        #. Create a graph
        #. Define ``is_training`` and ``global_step`` tensors
        #. Define a model architecture by calling :meth:``._build``
        #. Create a loss function from config
        #. Create an optimizer and define a train step from config
        #. `Set UPDATE_OPS control dependency on train step
           <https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization>`_
        #. Create a tensorflow session
        """
        with self.graph.as_default():
            with tf.variable_scope(self.__class__.__name__):
                with tf.variable_scope('globals'):
                    self.store_to_attr('is_training', tf.placeholder(tf.bool, name='is_training'))
                    self.store_to_attr('global_step', tf.Variable(0, trainable=False, name='global_step'))

                config = self.build_config()
                self._build(config)

                self._make_loss(config)
                self.store_to_attr('loss', tf.losses.get_total_loss())

                if self.train_step is None:
                    optimizer = self._make_optimizer(config)

                    if optimizer:
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):
                            train_step = optimizer.minimize(self.loss, global_step=self.global_step)
                            self.store_to_attr('train_step', train_step)
                else:
                    self.store_to_attr('train_step', self.train_step)

            session_config = self.get('session', config, default={})
            self.session = tf.Session(**session_config)
            self.session.run(tf.global_variables_initializer())

    def _make_inputs(self, names=None, config=None):
        """ Create model input data from config provided

        In the config's inputs section it looks for ``names``, creates placeholders required, and
        makes some typical transformations (like one-hot-encoding), if needed.

        **Configuration**

        inputs : dict
            - key : str
                a placeholder name
            - values : dict or tuple
                each input's config

        Input config:

        ``dtype`` : str or tf.DType (by default 'float32')
            data type

        ``shape`` : int, tuple, list or None (default)
            a tensor shape which includes the number of channels/classes and doesn't include a batch size.

        ``classes`` : array-like or None (default)
            an array of class labels if data labels are strings or anything else except ``np.arange(num_classes)``

        ``data_format`` : str {``'channels_first'``, ``'channels_last'``} or {``'f'``, ``'l'``}
            The ordering of the dimensions in the inputs. Default is 'channels_last'.

        ``transform`` : str or callable
            Predefined transforms are

            - ``'ohe'`` - one-hot encoding
            - ``'mip @ d'`` - maximum intensity projection :func:`~.layers.mip`
              with depth ``d`` (should be int)

        ``name`` : str
            a name for the transformed and reshaped tensor.

        If an input config is a tuple, it should contain all items exactly in the order shown above:
        dtype, shape, classes, data_format, transform, name.
        If an item is None, the default value will be used instead.

        **How it works**

        A placholder with ``dtype``, ``shape`` and with a name ``key`` is created first.
        Then it is transformed with a ``transform`` function in accordance with ``data_format``.
        The resulting tensor will have the name ``name``.

        Parameters
        ----------
        names : list
            placeholder names that are expected in the config's 'inputs' section

        Raises
        ------
        KeyError if there is any name missing in the config's 'inputs' section.
        ValueError if there are duplicate names.

        Returns
        -------
        placeholders : dict
            key : str
                a placeholder name
            value : tf.Tensor
                placeholder tensor
        tensors : dict
            key : str
                a placeholder name
            value : tf.Tensor
                an input tensor after transformations
        """
        # pylint:disable=too-many-statements
        config = self.get('inputs', config)

        names = names or []
        missing_names = set(names) - set(config.keys())
        if len(missing_names) > 0:
            raise KeyError("Inputs should contain {} names".format(missing_names))

        placeholder_names = set(config.keys())
        tensor_names = set(x.get('name') for x in config.values() if x.get('name'))
        wrong_names = placeholder_names & tensor_names
        if len(wrong_names) > 0:
            raise ValueError('Inputs contain duplicate names:', wrong_names)

        param_names = ('dtype', 'shape', 'classes', 'data_format', 'transform', 'name')
        defaults = dict(data_format='channels_last')

        placeholders = dict()
        tensors = dict()

        for input_name, input_config in config.items():
            if isinstance(input_config, (tuple, list)):
                input_config = list(input_config) + [None for _ in param_names]
                input_config = input_config[:len(param_names)]
                input_config = dict(zip(param_names, input_config))
                input_config = dict((k, v) for k, v in input_config.items() if v is not None)
            input_config = {**defaults, **input_config}

            reshape = None
            shape = input_config.get('shape')
            if isinstance(shape, int):
                shape = (shape,)
            if shape:
                input_config['shape'] = shape
                shape = [None] + list(shape)

            self._inputs[input_name] = dict(config=input_config)

            if self.has_classes(input_name):
                dtype = input_config.get('dtype', tf.int64)
                shape = shape or (None,)
            else:
                dtype = input_config.get('dtype', 'float')
            tensor = tf.placeholder(dtype, shape, input_name)
            placeholders[input_name] = tensor
            self.store_to_attr(input_name, tensor)

            if input_config.get('data_format') == 'l':
                input_config['data_format'] = 'channels_last'
            elif input_config.get('data_format') == 'f':
                input_config['data_format'] = 'channels_first'

            self._inputs[input_name] = dict(config=input_config)
            tensor = self._make_transform(input_name, tensor, input_config)

            if isinstance(reshape, (list, tuple)):
                tensor = tf.reshape(tensor, [-1] + list(reshape))

            name = input_config.get('name')
            if name is not None:
                tensor = tf.identity(tensor, name=name)
                self.store_to_attr(name, tensor)

            tensors[input_name] = tensor

            self._inputs[input_name] = dict(config=input_config, placeholder=placeholders[input_name], tensor=tensor)
            if name is not None:
                self._inputs[name] = self._inputs[input_name]

        self.inputs = tensors

        return placeholders, tensors

    def _make_transform(self, input_name, tensor, config):
        if config is not None:
            transform_name = config.get('transform')
            if isinstance(transform_name, str):
                transforms = {
                    'ohe': self._make_ohe,
                    'mip': self._make_mip
                }

                kwargs = dict()
                if transform_name.startswith('mip'):
                    parts = transform_name.split('@')
                    transform_name = parts[0].strip()
                    kwargs['depth'] = int(parts[1])

                tensor = transforms[transform_name](input_name, tensor, config, **kwargs)
            elif callable(transform_name):
                tensor = transform_name(tensor)
            elif transform_name:
                raise ValueError("Unknown transform {}".format(transform_name))
        return tensor

    def _make_ohe(self, input_name, tensor, config):
        if config.get('shape') is None and config.get('classes') is None:
            raise ValueError("shape and classes cannot be both None for input " +
                             "'{}' with one-hot-encoding transform".format(input_name))

        num_classes = self.num_classes(input_name)
        axis = -1 if self.data_format(input_name) else 1
        tensor = tf.one_hot(tensor, depth=num_classes, axis=axis)
        return tensor

    def to_classes(self, tensor, input_name, name=None):
        """ Convert tensor with labels to classes of ``input_name`` """
        if tensor.dtype in [tf.float16, tf.float32, tf.float64]:
            tensor = tf.argmax(tensor, axis=-1, name=name)
        if self.has_classes(input_name):
            self._to_classes.update({tensor: input_name})
        return tensor

    def _make_mip(self, input_name, tensor, config, depth):
        # mip has to know shape
        if config.get('shape') is None:
            raise ValueError('mip transform requires shape specified in the inputs config')
        if depth is None:
            raise ValueError("mip should be specified as mip @ depth, e.g. 'mip @ 3'")
        tensor = mip(tensor, depth=depth, data_format=self.data_format(input_name))
        return tensor

    def _unpack_fn_from_config(self, param, config=None):
        par = self.get(param, config)

        if par is None:
            return None, None

        if isinstance(par, (tuple, list)):
            if len(par) == 0:
                par_name = None
            elif len(par) == 1:
                par_name, par_args = par[0], {}
            elif len(par) == 2:
                par_name, par_args = par
            else:
                par_name, par_args = par[0], par[1:]
        elif isinstance(par, dict):
            par = par.copy()
            par_name, par_args = par.pop('name', None), par
        else:
            par_name, par_args = par, {}

        return par_name, par_args

    def _make_loss(self, config):
        """ Return a loss function from config """
        loss = self.get('loss', config)

        if isinstance(loss, dict):
            target_scope = loss.get('targets_scope', 'inputs')
            target_scope = '/' + target_scope + '/' if target_scope else ''
            prediction_scope = loss.get('predictions_scope', '')
            prediction_scope = '/' + prediction_scope + '/' if prediction_scope else ''
            loss = loss.get('loss')
        else:
            target_scope = 'inputs/'
            prediction_scope = ''

        if loss is None:
            if len(tf.losses.get_losses()) == 0:
                raise ValueError("Loss is not defined in the model %s" % self)
        elif isinstance(loss, str) and hasattr(tf.losses, loss):
            loss = getattr(tf.losses, loss)
        elif isinstance(loss, str):
            loss = LOSSES.get(re.sub('[-_ ]', '', loss).lower(), None)
        elif callable(loss):
            pass
        else:
            raise ValueError("Unknown loss", loss)

        if loss is not None:
            try:
                scope = self.graph.get_name_scope()
                scope = scope + '/' if scope else ''
                predictions = self.graph.get_tensor_by_name(scope + prediction_scope + "predictions:0")
                targets = self.graph.get_tensor_by_name(scope + target_scope + "targets:0")
            except KeyError:
                raise KeyError("Model %s does not have 'predictions' or 'targets' tensors" % self.name)
            else:
                tf.losses.add_loss(loss(targets, predictions))

    def _make_decay(self, config):
        decay_name, decay_args = self._unpack_fn_from_config('decay', config)

        if decay_name is None:
            pass
        elif callable(decay_name):
            pass
        elif isinstance(decay_name, str) and hasattr(tf.train, decay_name):
            decay_name = getattr(tf.train, decay_name)
        elif decay_name in DECAYS:
            decay_name = DECAYS.get(re.sub('[-_ ]', '', decay_name).lower(), None)
        else:
            raise ValueError("Unknown learning rate decay method", decay_name)

        return decay_name, decay_args

    def _make_optimizer(self, config):
        optimizer_name, optimizer_args = self._unpack_fn_from_config('optimizer', config)

        if optimizer_name is None or callable(optimizer_name):
            pass
        elif isinstance(optimizer_name, str) and hasattr(tf.train, optimizer_name):
            optimizer_name = getattr(tf.train, optimizer_name)
        elif isinstance(optimizer_name, str) and hasattr(tf.train, optimizer_name + 'Optimizer'):
            optimizer_name = getattr(tf.train, optimizer_name + 'Optimizer')
        else:
            raise ValueError("Unknown optimizer", optimizer_name)

        decay_name, decay_args = self._make_decay(config)
        if decay_name is not None:
            optimizer_args['learning_rate'] = decay_name(**decay_args, global_step=self.global_step)

        if optimizer_name:
            optimizer = optimizer_name(**optimizer_args)
        else:
            optimizer = None

        return optimizer

    def get_number_of_trainable_vars(self):
        """ Return the number of trainable variable in the model graph """
        with self.graph:
            arr = np.asarray([np.prod(self.get_shape(v)) for v in tf.trainable_variables()])
        return np.sum(arr)

    def get_tensor_config(self, tensor, **kwargs):
        """ Return tensor configuration

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        dict
            tensor config (see :meth:`._make_inputs`)

        Raises
        ------
        ValueError shape in tensor configuration isn't int, tuple or list
        """
        if isinstance(tensor, tf.Tensor):
            names = [n for n, i in self._inputs.items() if tensor in [i['placeholder'], i['tensor']]]
            if len(names) > 0:
                input_name = names[0]
            else:
                input_name = tensor.name
        elif isinstance(tensor, str):
            if tensor in self._inputs:
                input_name = tensor
            else:
                input_name = self._map_name(tensor)
        else:
            raise TypeError("Tensor can be tf.Tensor or string, but given %s" % type(tensor))

        if input_name in self._inputs:
            config = self._inputs[input_name]['config']
            shape = config.get('shape')
            if isinstance(shape, int):
                shape = (shape,)
            if shape:
                kwargs['shape'] = shape
        elif isinstance(input_name, str):
            try:
                tensor = self.graph.get_tensor_by_name(input_name)
            except KeyError:
                config = {}
            else:
                shape = tensor.get_shape().as_list()[1:]
                config = dict(dtype=tensor.dtype, shape=shape, name=tensor.name, data_format='channels_last')
        else:
            config = {}

        config = {**config, **kwargs}
        return config


    @staticmethod
    def channels_axis(data_format):
        """ Return the channels axis for the tensor

        Parameters
        ----------
        data_format : str {'channels_last', 'channels_first'}

        Returns
        -------
        number of channels : int
        """
        return -1 if data_format == "channels_last" or not data_format.startswith("NC") else 0

    def num_channels(self, tensor, **kwargs):
        """ Return the number of channels in the tensor

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        number of channels : int
        """
        config = self.get_tensor_config(tensor, **kwargs)
        shape = config.get('shape')
        channels_axis = self.channels_axis(tensor, **kwargs)
        return shape[channels_axis] if shape else None

    def has_classes(self, tensor):
        """ Check if a tensor has classes defined in the config """
        config = self.get_tensor_config(tensor)
        has = config.get('classes') is not None
        return has

    def classes(self, tensor):
        """ Return the  number of classes """
        config = self.get_tensor_config(tensor)
        classes = config.get('classes')
        if isinstance(classes, int):
            return np.arange(classes)
        return classes

    def num_classes(self, tensor):
        """ Return the  number of classes """
        if self.has_classes(tensor):
            classes = self.classes(tensor)
            return classes if isinstance(classes, int) else len(classes)
        return self.num_channels(tensor)

    def spatial_dim(self, tensor, **kwargs):
        """ Return the tensor spatial dimensionality (without channels dimension)

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        number of spatial dimensions : int
        """
        config = self.get_tensor_config(tensor, **kwargs)
        return len(config.get('shape')) - 1

    def data_format(self, tensor, **kwargs):
        """ Return the tensor data format (channels_last or channels_first)

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        data_format : str
        """
        config = self.get_tensor_config(tensor, **kwargs)
        return config.get('data_format')

    @staticmethod
    def batch_size(tensor):
        """ Return batch size (the length of the first dimension) of the input tensor

        Parameters
        ----------
        tensor : tf.Tensor

        Returns
        -------
        batch size : int
        """
        return tensor.get_shape().as_list()[0]

    @staticmethod
    def get_shape(tensor):
        """ Return full shape of the input tensor

        Parameters
        ----------
        tensor : tf.Tensor

        Returns
        -------
        shape : tuple of ints
        """
        return tensor.get_shape().as_list()

    @staticmethod
    def spatial_shape(tensor, data_format='channels_last'):
        """ Return spatial shape of the input tensor

        Parameters
        ----------
        tensor : tf.Tensor

        Returns
        -------
        shape : tuple of ints
        """
        shape = tensor.get_shape().as_list()
        axis = slice(1, -1) if data_format == "channels_last" else slice(2, None)
        return shape[axis]

    @staticmethod
    def channels_shape(tensor, data_format='channels_last'):
        """ Return number of channels in the input tensor

        Parameters
        ----------
        tensor : tf.Tensor

        Returns
        -------
        shape : tuple of ints
        """
        shape = tensor.get_shape().as_list()
        axis = TFModel.channels_axis(data_format)
        return shape[axis]

    def _map_name(self, name):
        if isinstance(name, str):
            if hasattr(self, name):
                return getattr(self, name)
            elif ':' in name:
                return name
            return name + ':0'
        return name

    def _fill_feed_dict(self, feed_dict=None, is_training=True):
        feed_dict = feed_dict or {}
        _feed_dict = {}
        for placeholder, value in feed_dict.items():
            if self.has_classes(placeholder):
                classes = self.classes(placeholder)
                get_indices = np.vectorize(lambda c, arr=classes: np.where(c == arr)[0])
                value = get_indices(value)
            placeholder = self._map_name(placeholder)
            value = self._map_name(value)
            _feed_dict.update({placeholder: value})
        if self.is_training not in _feed_dict:
            _feed_dict.update({self.is_training: is_training})
        return _feed_dict

    def _fill_fetches(self, fetches=None, default=None):
        fetches = fetches or default
        if isinstance(fetches, str):
            _fetches = self._map_name(fetches)
        elif isinstance(fetches, (tuple, list)):
            _fetches = []
            for fetch in fetches:
                _fetches.append(self._map_name(fetch))
        elif isinstance(fetches, dict):
            _fetches = dict()
            for key, fetch in fetches.items():
                _fetches.update({key: self._map_name(fetch)})
        else:
            _fetches = fetches
        return _fetches

    def _fill_output(self, output, fetches):
        def _recast_output(out, ix=None):
            if isinstance(out, np.ndarray):
                fetch = fetches[ix] if ix is not None else fetches
                if isinstance(fetch, str):
                    fetch = self.graph.get_tensor_by_name(fetch)
                if fetch in self._to_classes:
                    return self.classes(self._to_classes[fetch])[out]
            return out

        if isinstance(output, (tuple, list)):
            _output = []
            for i, o in enumerate(output):
                _output.append(_recast_output(o, i))
            output = type(output)(_output)
        elif isinstance(output, dict):
            _output = type(output)()
            for k, v in output.items():
                _output.update({k: _recast_output(v, k)})
        else:
            output = _recast_output(output)

        return output

    def train(self, fetches=None, feed_dict=None, use_lock=False):   # pylint: disable=arguments-differ
        """ Train the model with the data provided

        Parameters
        ----------
        fetches : tuple, list
            a sequence of `tf.Operation` and/or `tf.Tensor` to calculate
        feed_dict : dict
            input data, where key is a placeholder name and value is a numpy value
        use_lock : bool
            if True, the whole train step is locked, thus allowing for multithreading.

        Returns
        -------
        Calculated values of tensors in `fetches` in the same structure

        See also
        --------
        `Tensorflow Session run <https://www.tensorflow.org/api_docs/python/tf/Session#run>`_
        """
        with self.graph.as_default():
            _feed_dict = self._fill_feed_dict(feed_dict, is_training=True)
            if fetches is None:
                _fetches = tuple()
            else:
                _fetches = self._fill_fetches(fetches, default=None)

            if use_lock:
                self._train_lock.acquire()

            _all_fetches = []
            if self.train_step:
                _all_fetches += [self.train_step]
            if _fetches is not None:
                _all_fetches += [_fetches]
            if len(_all_fetches) > 0:
                _, output = self.session.run(_all_fetches, feed_dict=_feed_dict)
            else:
                output = None

            if use_lock:
                self._train_lock.release()

            return self._fill_output(output, _fetches)

    def predict(self, fetches=None, feed_dict=None):      # pylint: disable=arguments-differ
        """ Get predictions on the data provided

        Parameters
        ----------
        fetches : tuple, list
            a sequence of `tf.Operation` and/or `tf.Tensor` to calculate
        feed_dict : dict
            input data, where key is a placeholder name and value is a numpy value

        Returns
        -------
        Calculated values of tensors in `fetches` in the same structure

        Notes
        -----
        The only difference between `predict` and `train` is that `train` also executes a `train_step` operation
        which involves calculating and applying gradients and thus chainging model weights.

        See also
        --------
        `Tensorflow Session run <https://www.tensorflow.org/api_docs/python/tf/Session#run>`_
        """
        with self.graph.as_default():
            _feed_dict = self._fill_feed_dict(feed_dict, is_training=False)
            _fetches = self._fill_fetches(fetches, default='predictions')
            output = self.session.run(_fetches, _feed_dict)
        return self._fill_output(output, _fetches)

    def save(self, path, *args, **kwargs):
        """ Save tensorflow model.

        Parameters
        ----------
        path : str
            a path to a directory where all model files will be stored

        Examples
        --------
        >>> tf_model = ResNet34()

        Now save the model

        >>> tf_model.save('/path/to/models/resnet34')

        The model will be saved to /path/to/models/resnet34
        """
        with self.graph.as_default():
            if not os.path.exists(path):
                os.makedirs(path)
            saver = tf.train.Saver()
            saver.save(self.session, os.path.join(path, 'model'), *args, global_step=self.global_step, **kwargs)
            with open(os.path.join(path, 'attrs.json'), 'w') as f:
                json.dump(self._attrs, f)

    def load(self, path, graph=None, checkpoint=None, *args, **kwargs):
        """ Load a tensorflow model from files

        Parameters
        ----------
        path : str
            a directory where a model is stored
        graph : str
            a filename for a metagraph file
        checkpoint : str
            a checkpoint file name or None to load the latest checkpoint

        Examples
        --------
        >>> tf_model = ResNet34(load=True)

        >>> tf_model.load('/path/to/models/resnet34')
        """
        _ = args, kwargs
        self.session = tf.Session()

        with self.session.as_default():
            if graph is None:
                graph_files = glob.glob(os.path.join(path, '*.meta'))
                graph_files = [os.path.splitext(os.path.basename(graph))[0] for graph in graph_files]
                all_steps = []
                for graph in graph_files:
                    try:
                        step = int(graph.split('-')[-1])
                    except ValueError:
                        pass
                    else:
                        all_steps.append(step)
                graph = '-'.join(['model', str(max(all_steps))]) + '.meta'

            graph_path = os.path.join(path, graph)
            saver = tf.train.import_meta_graph(graph_path)

            if checkpoint is None:
                checkpoint_path = tf.train.latest_checkpoint(path)
            else:
                checkpoint_path = os.path.join(path, checkpoint)

            saver.restore(self.session, checkpoint_path)
            self.graph = self.session.graph

        with open(os.path.join(path, 'attrs.json'), 'r') as json_file:
            self._attrs = json.load(json_file)
        with self.graph.as_default():
            for attr, graph_item in zip(self._attrs, tf.get_collection('attrs')):
                setattr(self, attr, graph_item)

    def store_to_attr(self, attr, graph_item):
        """ Make a graph item (variable or operation) accessible as a model attribute """
        with self.graph.as_default():
            setattr(self, attr, graph_item)
            self._attrs.append(attr)
            tf.get_collection_ref('attrs').append(graph_item)

    @classmethod
    def crop(cls, inputs, shape_images, data_format='channels_last'):
        """ Crop input tensor to a shape of a given image.
        If shape_image has not fully defined shape (shape_image.get_shape() has at leats one None),
        the returned tf.Tensor will be of unknown shape except the number of channels.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        shape_images : tf.Tensor
            a source images to which
        data_format : str {'channels_last', 'channels_first'}
            data format
        """
        axis = slice(1, -1) if data_format == 'channels_last' else slice(2, None)
        static_shape = shape_images.get_shape().as_list()[axis]
        dynamic_shape = tf.shape(shape_images)[axis]

        if None in inputs.get_shape().as_list()[1:] + static_shape:
            return cls._dynamic_crop(inputs, static_shape, dynamic_shape, data_format)
        else:
            return cls._static_crop(inputs, static_shape, data_format)

    @classmethod
    def _static_crop(cls, inputs, shape, data_format='channels_last'):
        input_shape = np.array(cls.spatial_shape(inputs, data_format))

        if np.abs(input_shape - shape).sum() > 0:
            begin = [0] * inputs.shape.ndims
            if data_format == "channels_last":
                size = [-1] + shape + [-1]
            else:
                size = [-1, -1] + shape
            x = tf.slice(inputs, begin=begin, size=size)
        else:
            x = inputs
        return x

    @classmethod
    def _dynamic_crop(cls, inputs, static_shape, dynamic_shape, data_format='channels_last'):
        if data_format == 'channels_last':
            input_shape = tf.shape(inputs)[1:-1]
            n_channels = inputs.get_shape().as_list()[-1]
            slice_size = [(-1,), dynamic_shape, (n_channels,)]
            output_shape = [None] * (len(static_shape) + 1) + [n_channels]
        else:
            input_shape = tf.shape(inputs)[2:]
            n_channels = inputs.get_shape().as_list()[1]
            slice_size = [(-1, n_channels), dynamic_shape]
            output_shape = [None, n_channels] + [None] * len(static_shape)

        begin = [0] * len(inputs.get_shape().as_list())
        size = tf.concat(slice_size, axis=0)
        cond = tf.reduce_sum(tf.abs(input_shape - dynamic_shape)) > 0
        x = tf.cond(cond, lambda: tf.slice(inputs, begin=begin, size=size), lambda: inputs)
        x.set_shape(output_shape)
        return x

    @classmethod
    def input_block(cls, inputs, name='input_block', **kwargs):
        """ Transform inputs with a convolution block

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Notes
        -----
        For other parameters see :func:`.conv_block`.

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('input_block', **kwargs)
        if kwargs.get('layout'):
            return conv_block(inputs, name=name, **kwargs)
        return inputs

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers which produce a network embedding

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Notes
        -----
        For other parameters see :func:`.conv_block`.

        Returns
        -------
        tf.Tensor

        Examples
        --------
        ::

            MyModel.body(2, inputs, layout='ca ca ca', filters=[128, 256, 512], kernel_size=3)
        """
        kwargs = cls.fill_params('body', **kwargs)
        if kwargs.get('layout'):
            return conv_block(inputs, name=name, **kwargs)
        return inputs

    @classmethod
    def head(cls, inputs, name='head', **kwargs):
        """ Last network layers which produce output from the network embedding

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Notes
        -----
        For other parameters see :func:`.conv_block`.

        Returns
        -------
        tf.Tensor

        Examples
        --------
        A fully convolutional head with 3x3 and 1x1 convolutions and global max pooling:

            MyModel.head(2, network_embedding, layout='cacaP', filters=[128, num_classes], kernel_size=[3, 1])

        A fully connected head with dropouts, a dense layer with 1000 units and final dense layer with class logits::

            MyModel.head(2, network_embedding, layout='dfadf', units=[1000, num_classes], dropout_rate=.15)
        """
        kwargs = cls.fill_params('head', **kwargs)
        if kwargs.get('layout'):
            return conv_block(inputs, name=name, **kwargs)
        return inputs

    def output(self, inputs, ops=None, prefix=None, **kwargs):
        """ Add output operations to a model graph, like predictions, quality metrics, etc.

        Parameters
        ----------
        inputs : tf.Tensor or a sequence of tf.Tensors
            input tensors

        ops : a sequence of str
            operation names::
            - 'proba' - add ``softmax(inputs)``
            - 'labels' - add ``argmax(inputs)``
            - 'accuracy' - add ``mean(predicted_labels == true_labels)``

        prefix : a sequence of str
            a prefix for each input if there are multiple inputs

        Raises
        ------
        ValueError if the number of outputs does not equal to the number of prefixes
        TypeError if inputs is not a Tensor or a sequence of Tensors
        """
        kwargs = self.fill_params('output', **kwargs)

        if ops is None:
            ops = []
        elif not isinstance(ops, (list, tuple)):
            ops = [ops]

        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
            prefix = [prefix]

        if len(inputs) != len(prefix):
            raise ValueError('Each output in multiple output models should have its own prefix')

        for i, tensor in enumerate(inputs):
            if not isinstance(tensor, tf.Tensor):
                raise TypeError("Network output is expected to be a Tensor, but given {}".format(type(inputs)))

            scope = self.graph.get_name_scope()
            scope = scope + '/' if scope else ''
            current_prefix = prefix[i] or ''
            if current_prefix:
                ctx = tf.variable_scope(current_prefix)
                ctx.__enter__()
            else:
                ctx = None
            attr_prefix = current_prefix + '_' if current_prefix else ''

            x = self._add_output_predictions(tensor, scope, attr_prefix, **kwargs)
            for oper in ops:
                if oper == 'proba':
                    self._add_output_proba(x, scope, attr_prefix, **kwargs)
                elif oper == 'labels':
                    self._add_output_labels(x, scope, attr_prefix, **kwargs)
                elif oper == 'accuracy':
                    self._add_output_accuracy(x, scope, attr_prefix, **kwargs)

            if ctx:
                ctx.__exit__(None, None, None)

    def _add_output_predictions(self, inputs, scope, attr_prefix, **kwargs):
        _ = scope, kwargs
        x = tf.identity(inputs, name='predictions')
        self.store_to_attr(attr_prefix + 'predictions', x)
        return x

    def _add_output_proba(self, inputs, scope, attr_prefix, **kwargs):
        _ = scope, kwargs
        proba = tf.nn.softmax(inputs, name='predicted_proba')
        self.store_to_attr(attr_prefix + 'predicted_proba', proba)

    def _add_output_labels(self, inputs, scope, attr_prefix, **kwargs):
        _ = scope
        data_format = kwargs.get('data_format')
        channels_axis = self.channels_axis(data_format)
        predicted_labels = tf.argmax(inputs, axis=channels_axis, name='predicted_labels')
        self.store_to_attr(attr_prefix + 'predicted_labels', predicted_labels)

    def _add_output_accuracy(self, inputs, scope, attr_prefix, **kwargs):
        true_labels = self.graph.get_tensor_by_name(scope + 'inputs/labels:0')
        current_scope = self.graph.get_name_scope() + '/'
        try:
            predicted_labels = self.graph.get_tensor_by_name(current_scope + 'predicted_labels:0')
        except KeyError:
            self._add_output_labels(inputs, scope, attr_prefix, **kwargs)
            predicted_labels = self.graph.get_tensor_by_name(current_scope + 'predicted_labels:0')
        equals = tf.cast(tf.equal(true_labels, predicted_labels), 'float')
        accuracy = tf.reduce_mean(equals, name='accuracy')
        self.store_to_attr(attr_prefix + 'accuracy', accuracy)


    @classmethod
    def default_config(cls):
        """ Define model defaults

        You need to override this method if you expect your model or its blocks to serve as a base for other models
        (e.g. VGG for FCN, ResNet for LinkNet, etc).

        Put here all constants (like the number of filters, kernel sizes, block layouts, strides, etc)
        specific to the model, but independent of anything else (like image shapes, number of classes, etc).

        These defaults can be changed in `.build_config` or when calling `Pipeline.init_model`.

        Usually, it looks like::

            @classmethod
            def default_config(cls):
                config = TFModel.default_config()
                config['input_block'].update(dict(layout='cnap', filters=16, kernel_size=7, strides=2,
                                                  pool_size=3, pool_strides=2))
                config['body']['filters'] = 32
                config['head'].update(dict(layout='cnadV', dropout_rate=.2))
                return config
        """
        config = {}
        config['inputs'] = {}
        config['common'] = {'batch_norm': {'momentum': .1}}
        config['input_block'] = {}
        config['body'] = {}
        config['head'] = {}
        config['output'] = {}
        config['loss'] = 'ce'
        config['optimizer'] = 'Adam'
        return config

    @classmethod
    def fill_params(cls, _name, **kwargs):
        """ Fill block params from default config and kwargs """
        config = cls.default_config()
        config = {**config['common'], **cls.get(_name, config), **kwargs}
        return config

    def build_config(self, names=None):
        """ Define a model architecture configuration

        It takes just 2 steps:

        #. Define names for all placeholders and make input tensors by calling ``super().build_config(names)``.

           If the model config does not contain any name from ``names``, :exc:`KeyError` is raised.

           See :meth:`._make_inputs` for details.

        #. Define parameters for :meth:`.TFModel.input_block`, :meth:`.TFModel.body`, :meth:`.TFModel.head`
           which depend on inputs.

        #. Don't forget to return ``config``.

        Typically it looks like this::

            def build_config(self, names=None):
                names = names or ['images', 'labels']
                config = super().build_config(names)
                config['head']['num_classes'] = self.num_classes('targets')
                return config
        """

        config = self.default_config()

        for k in self.config:
            self.put(k, self.config[k], config)

        if 'inputs' in self.config:
            with tf.variable_scope('inputs'):
                self._make_inputs(names, self.config)
            inputs = self.get('input_block/inputs', config)

            if isinstance(inputs, str):
                config['common']['data_format'] = self.data_format(inputs)
                config['input_block']['inputs'] = self.inputs[inputs]
            elif isinstance(inputs, list):
                config['input_block']['inputs'] = [self.inputs[name] for name in inputs]
            else:
                raise ValueError('input_block/inputs should be specified with a name or a list of names.')

        config['body'] = {**config['body'], **self.get('body', self.config, {})}
        config['head'] = {**config['head'], **self.get('head', self.config, {})}
        config['output'] = {**config['output'], **self.get('output', self.config, {})}
        return config


    def _build(self, config=None):
        defaults = {'is_training': self.is_training, **config['common']}
        config['input_block'] = {**defaults, **config['input_block']}
        config['body'] = {**defaults, **config['body']}
        config['head'] = {**defaults, **config['head']}
        config['output'] = {**defaults, **config['output']}

        x = self.input_block(**config['input_block'])
        x = self.body(inputs=x, **config['body'])
        output = self.head(inputs=x, **config['head'])
        self.output(output, **config['output'])

    @classmethod
    def se_block(cls, inputs, ratio, name='se', **kwargs):
        """ Squeeze and excitation block

        Hu J. et al. "`Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_"

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        ratio : int
            squeeze ratio for the number of filters

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            data_format = kwargs.get('data_format')
            in_filters = cls.channels_shape(inputs, data_format)
            x = conv_block(inputs, 'Vfafa', units=[in_filters//ratio, in_filters], name='se',
                           **{**kwargs, 'activation': [tf.nn.relu, tf.nn.sigmoid]})

            shape = [-1] + [1] * (len(cls.spatial_shape(inputs, data_format)) + 1)
            axis = cls.channels_axis(data_format)
            shape[axis] = in_filters
            scale = tf.reshape(x, shape)
            x = inputs * scale
        return x

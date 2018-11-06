# pylint: disable=super-init-not-called
# pylint: disable=not-context-manager
""" Contains base class for all keras models. """

import functools
import numpy as np
import tensorflow as tf
from keras.models import Model

from keras.layers import Flatten
from keras.layers import Dropout, Activation
from keras.layers import Dense, BatchNormalization

from ...batchflow.batchflow.models import BaseModel


class KerasModel(Model, BaseModel):
    """ Base class for all keras models.

    Contains `load`, `dump` and `compile` methods which are shared between all
    keras models. Implements train and predict methods.

    """
    def __init__(self, config=None, *args, **kwargs):
        """ Create keras model. """
        BaseModel.__init__(self, config, *args, **kwargs)

    def build_config(self):
        """ Build config. """
        input_shape = self.get('input_shape', self.config, (32, 64, 64, 1))
        num_targets = self.get('num_targets', self.config, 1)
        dropout_rate = self.get('dropout_rate', self.config, 0.35)
        units = self.get('units', self.config, (512, 256))
        if isinstance(units, int):
            units = (units, )
        elif units is None:
            units = ()
        self.config.update({'units': units, 'input_shape': input_shape,
                            'dropout_rate': dropout_rate,
                            'num_targets': num_targets})

    def build(self, *args, **kwargs):
        """ Must return inputs and outputs. """
        self.build_config()
        input_nodes, output_nodes = self._build()
        Model.__init__(self, input_nodes, output_nodes)
        self.compile(loss=self.get('loss', self.config, None),
                     optimizer=self.get('optimizer', self.config, 'sgd'))

    def _build(self, *args, **kwargs):
        """ Must return inputs and outputs. """
        raise NotImplementedError("This method must be implemented in ancestor model class")

    @classmethod
    def dense_block(cls, inputs, units, activation='relu', dropout=None, scope='DenseBlock'):
        """ Dense block for keras models.

        This block consists of flatten operation applied to inputs tensor.
        Then there is several fully connected layers with same activation,
        batch normalization and dropout layers. Usually this block is put
        in the end of the neural network.

        Parameters
        ----------
        inputs : keras tensor
            input tensor.
        units : tuple(int, ...)
            tuple with number of units in dense layers followed one by one.
        dropout : float or None
            probability of dropout.
        scope : str
            scope name for this block, will be used as an argument of tf.variable_scope.

        Returns:
        keras tensor
            output tensor.
        """
        with tf.variable_scope(scope):
            z = Flatten(name='flatten')(inputs)
            for u in units:
                z = Dense(u, name='Dense-{}'.format(u))(z)
                z = BatchNormalization(axis=-1)(z)
                z = Activation(activation)(z)
                if dropout is not None:
                    z = Dropout(dropout)(z)
        return z

    def train(self, x=None, y=None, **kwargs):
        """ Wrapper for keras.models.Model.train_on_batch.

        Parameters
        ----------
        x : ndarray(batch_size, ...)
            x argument of keras.models.Model.train_on_batch method, input of
            neural network.
        y : ndarray(batch_size, ...)
            y argument of keras.models.Model.predict_on_batch method.

        Returns
        -------
        ndarray(batch_size, ...)
            predictions of keras model.

        Raises
        ------
        ValueError if 'x' or 'y'  is None.
        """
        if x is None or y is None:
            raise ValueError("Arguments 'x' and 'y' must not be None")

        prediction = np.asarray(self.train_on_batch(x=x, y=y))
        return prediction

    def predict(self, x=None, **kwargs):
        """ Wrapper for keras.models.Model.predict_on_batch.

        Parameters
        ----------
        x : ndarray(batch_size, ...)
            x argument of keras.models.Model.predict_on_batch method, input of
            neural network.

        Returns
        -------
        ndarray(batch_size, ...)
            predictions of keras model.

        Raises
        ------
        ValueError if 'x' argument is None.
        """
        if x is not None:
            return Model.predict_on_batch(self, x=x)
        else:
            raise ValueError("Argument 'x' must not be None")
        return None

    @functools.wraps(Model.load_weights)
    def load(self, *args, **kwargs):
        """ Wrapper for keras.models.Model.load_weights. """
        return Model.load_weights(self, *args, **kwargs)

    @functools.wraps(Model.save_weights)
    def save(self, *args, **kwargs):
        """ Wrapper for keras.models.Model.save_weights. """
        return Model.save_weights(self, *args, **kwargs)
